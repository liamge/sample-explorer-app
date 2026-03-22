import math
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependencies. Install with: pip install -r requirements.txt"
    ) from e


AUDIO_EXTENSIONS = {".wav", ".aiff", ".aif", ".flac", ".ogg"}
MAX_ANALYSIS_SECONDS = 12  # limit per file for feature extraction
GRAPH_NEIGHBORS = 5
CACHE_NAME = ".sample_explorer_map.npz"
ACCENT = "#6ae3ff"
ACCENT_WARM = "#ffde7a"
BG_DARK = "#0b0d14"
VEL_THRESHOLD = 0.002
STABLE_TICKS_REQUIRED = 18
SIM_TIME_LIMIT = 1.2  # seconds


class SampleExplorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sample Cluster Explorer")
        self.root.geometry("1180x760")

        self.current_folder: Path | None = None
        self.sample_paths: list[Path] = []
        self.sample_meta: list[tuple[Path, float]] = []  # path, mtime
        self.current_path: Path | None = None

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.play_thread: threading.Thread | None = None
        self.is_playing = False
        self.is_building = False
        self.progress_total = 0
        self.progress_done = 0
        self._pulse_job: int | None = None

        self.cluster_force_var = tk.DoubleVar(value=0.9)
        self.damping_var = tk.DoubleVar(value=0.96)
        self.status_var = tk.StringVar(value="Choose a folder to begin.")
        self.info_var = tk.StringVar(value="")

        self.graph_coords: np.ndarray | None = None  # initial 2D projection
        self.graph_positions: np.ndarray | None = None  # layout after force sim
        self.graph_edges: list[tuple[int, int, float]] = []
        self.graph_node_paths: list[Path] = []
        self.graph_canvas_ids: dict[int, Path] = {}
        self._relayout_after_id: int | None = None
        self.layout_thread: threading.Thread | None = None
        self.sim_job: int | None = None
        self.velocities: np.ndarray | None = None
        self.screen_positions: list[tuple[float, float]] = []
        self._draw_toggle = False
        self.stable_ticks = 0
        self.last_stable_damping = None
        self.neighbor_lists: list[list[int]] = []
        self.sim_start_time: float | None = None

        self._build_ui()

    # --- UI ------------------------------------------------------------------------

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_DARK, foreground="#d7e2f2")
        style.configure("TButton", background="#161b28", foreground="#d7e2f2")
        style.configure("TScale", background=BG_DARK)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 10))

        title = ttk.Label(topbar, text="Sample Cluster Explorer", font=("Helvetica", 14, "bold"))
        title.pack(side="left", padx=(0, 18))

        ttk.Button(topbar, text="Open Folder", command=self.choose_folder).pack(side="left")
        ttk.Button(topbar, text="Rescan / Detect New", command=self.scan_folder).pack(side="left", padx=(8, 12))

        ttk.Label(topbar, text="Cluster force").pack(side="left")
        ttk.Scale(
            topbar,
            from_=0.4,
            to=2.4,
            variable=self.cluster_force_var,
            command=lambda *_: self._schedule_relayout(),
            length=200,
        ).pack(side="left", padx=(6, 12))
        ttk.Label(topbar, text="Damping").pack(side="left", padx=(0, 6))
        ttk.Scale(
            topbar,
            from_=0.75,
            to=0.99,
            variable=self.damping_var,
            command=lambda *_: self._on_damping_change(),
            length=160,
        ).pack(side="left", padx=(0, 12))
        ttk.Button(topbar, text="Re-run layout", command=self.relayout_graph).pack(side="left")

        ttk.Label(topbar, textvariable=self.status_var).pack(side="right")

        # Graph canvas fills most of the window
        self.graph_canvas = tk.Canvas(outer, bg=BG_DARK, highlightthickness=1)
        self.graph_canvas.pack(fill="both", expand=True)
        self.graph_canvas.bind("<Button-1>", self.on_graph_click)
        self.graph_canvas.bind("<Configure>", lambda e: self.draw_graph())

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(10, 0))
        ttk.Label(bottom, textvariable=self.info_var, anchor="w").pack(side="left", fill="x", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- Folder + scanning ----------------------------------------------------------

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select sample folder")
        if not folder:
            return
        self.current_folder = Path(folder)
        self.scan_folder()

    def scan_folder(self) -> None:
        if not self.current_folder:
            return
        self.sample_paths = sorted(
            [p for p in self.current_folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
        )
        self.sample_meta = [(p, p.stat().st_mtime) for p in self.sample_paths]
        self.status_var.set(f"Found {len(self.sample_paths)} files. Building map…")
        self._start_busy("Analyzing samples…")
        self._start_projection_thread()

    # --- Projection + layout --------------------------------------------------------

    def _start_projection_thread(self) -> None:
        if self.layout_thread and self.layout_thread.is_alive():
            return
        self.layout_thread = threading.Thread(target=self._compute_projection, daemon=True)
        self.layout_thread.start()

    def _compute_projection(self) -> None:
        cache = self._load_cache()
        cached_vectors: dict[Path, np.ndarray] = {}
        if cache:
            for p_str, mtime, vec in zip(cache["paths"], cache["mtimes"], cache["features"]):
                p = Path(p_str)
                if not p.exists():
                    continue
                current_mtime = next((m for path, m in self.sample_meta if path == p), None)
                if current_mtime is None or abs(current_mtime - mtime) > 1e-3:
                    continue
                cached_vectors[p] = vec

        vectors: list[np.ndarray] = []
        node_paths: list[Path] = []
        self.progress_total = len(self.sample_paths)
        self.progress_done = 0

        for path in self.sample_paths:
            vec = cached_vectors.get(path)
            if vec is None:
                vec = self._feature_vector_for(path)
            if vec is None:
                continue
            vectors.append(vec)
            node_paths.append(path)
            self.progress_done += 1
            self._queue_progress()

        if not vectors:
            self.root.after(0, lambda: self.status_var.set("No analyzable audio files."))
            return

        X = np.vstack(vectors)
        coords = self._pca_2d(X)
        edges = self._build_edges(X, k=GRAPH_NEIGHBORS)
        positions = self._force_layout(coords.copy(), edges, force=self.cluster_force_var.get())

        def _store() -> None:
            self.graph_node_paths = node_paths
            self.graph_coords = coords
            self.graph_positions = positions
            self.velocities = np.zeros_like(positions)
            self.graph_edges = edges
            self.neighbor_lists = self._build_neighbor_lists(len(positions), edges)
            self._save_cache(node_paths, vectors, coords, positions, edges)
            self._stop_busy()
            self.draw_graph()
            self.status_var.set("Map ready — click a node to audition")
            self._start_physics()

        self.root.after(0, _store)

    def _feature_vector_for(self, path: Path) -> np.ndarray | None:
        try:
            audio, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception:
            return None

        if len(audio) == 0:
            return None

        mono = audio.mean(axis=1)
        limit = min(len(mono), MAX_ANALYSIS_SECONDS * sr)
        mono = mono[:limit]
        if len(mono) == 0:
            return None

        rms = float(np.sqrt(np.mean(mono**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(mono)))) / 2.0)

        spec = np.abs(np.fft.rfft(mono))
        power = spec**2 + 1e-12
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
        centroid = float(np.sum(freqs * power) / np.sum(power))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power)))
        cumulative = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        band_edges = np.geomspace(30, sr / 2, num=9)
        band_energy = []
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            band_energy.append(float(power[mask].mean() if mask.any() else 0.0))

        features = np.array([rms, zcr, centroid, bandwidth, rolloff, *band_energy], dtype=np.float32)
        norm = np.linalg.norm(features) or 1.0
        return features / norm

    def _pca_2d(self, X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        coords = Xc @ vt[:2].T
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.where(maxs - mins == 0, 1.0, maxs - mins)
        return (coords - mins) / span

    def _cache_path(self) -> Path:
        if not self.current_folder:
            return Path(CACHE_NAME)
        return self.current_folder / CACHE_NAME

    def _save_cache(
        self,
        paths: list[Path],
        vectors: list[np.ndarray],
        coords: np.ndarray,
        positions: np.ndarray,
        edges: list[tuple[int, int, float]],
    ) -> None:
        try:
            np.savez(
                self._cache_path(),
                paths=np.array([str(p) for p in paths]),
                mtimes=np.array([p.stat().st_mtime for p in paths], dtype=float),
                features=np.vstack(vectors),
                coords=coords,
                positions=positions,
                edges=np.array(edges, dtype=float),
            )
        except Exception:
            # cache failures are non-fatal
            pass

    def _load_cache(self) -> dict | None:
        try:
            cache_path = self._cache_path()
            if not cache_path.exists():
                return None
            data = np.load(cache_path, allow_pickle=True)
            return {
                "paths": data["paths"],
                "mtimes": data["mtimes"],
                "features": data["features"],
                "coords": data.get("coords"),
                "positions": data.get("positions"),
                "edges": data.get("edges"),
            }
        except Exception:
            return None

    # --- Busy / progress UI ---------------------------------------------------------

    def _start_busy(self, message: str) -> None:
        self.is_building = True
        self.progress_done = 0
        self.progress_total = max(1, len(self.sample_paths))
        self.busy_message = message
        self._draw_overlay(message)
        self._pulse_overlay()

    def _stop_busy(self) -> None:
        self.is_building = False
        if self._pulse_job:
            self.root.after_cancel(self._pulse_job)
            self._pulse_job = None
        self._clear_overlay()
        # restart physics if a map exists
        self._start_physics()

    def _queue_progress(self) -> None:
        if not self.is_building:
            return
        frac = self.progress_done / max(1, self.progress_total)
        msg = f"{self.busy_message}  {self.progress_done}/{self.progress_total}"
        self.root.after(0, lambda: self._draw_overlay(msg, frac))

    def _draw_overlay(self, msg: str, frac: float | None = None) -> None:
        self._clear_overlay()
        w = max(self.graph_canvas.winfo_width(), 200)
        h = max(self.graph_canvas.winfo_height(), 200)
        overlay = self.graph_canvas.create_rectangle(0, 0, w, h, fill="#0b0d14", stipple="gray25", outline="")
        text = self.graph_canvas.create_text(
            w / 2,
            h / 2 - 10,
            text=msg,
            fill="#d7e2f2",
            font=("Helvetica", 12, "bold"),
        )
        bar_bg = self.graph_canvas.create_rectangle(w / 2 - 120, h / 2 + 12, w / 2 + 120, h / 2 + 30, fill="#1c2333")
        if frac is None:
            frac = 0.08
        bar_fg = self.graph_canvas.create_rectangle(
            w / 2 - 118,
            h / 2 + 14,
            w / 2 - 118 + 236 * max(0.02, min(frac, 1.0)),
            h / 2 + 28,
            fill=ACCENT,
            outline="",
        )
        self.overlay_items = [overlay, text, bar_bg, bar_fg]

    def _clear_overlay(self) -> None:
        if hasattr(self, "overlay_items"):
            for item in self.overlay_items:
                try:
                    self.graph_canvas.delete(item)
                except Exception:
                    pass
            self.overlay_items = []

    def _pulse_overlay(self) -> None:
        if not self.is_building:
            return
        # Keep overlay visible; redraw it to pulse progress bar gently.
        frac = self.progress_done / max(1, self.progress_total)
        self._draw_overlay(self.busy_message, frac)
        self._pulse_job = self.root.after(420, self._pulse_overlay)

    def _build_edges(self, X: np.ndarray, k: int) -> list[tuple[int, int, float]]:
        norms = np.linalg.norm(X, axis=1) + 1e-9
        sims = (X @ X.T) / np.outer(norms, norms)
        edges: set[tuple[int, int, float]] = set()
        for i in range(len(X)):
            neighbors = np.argsort(-sims[i])
            count = 0
            for j in neighbors:
                if i == j:
                    continue
                edges.add((min(i, j), max(i, j), float(sims[i, j])))
                count += 1
                if count >= k:
                    break
        return sorted(list(edges), key=lambda e: e[2], reverse=True)

    def _build_neighbor_lists(self, n: int, edges: list[tuple[int, int, float]]) -> list[list[int]]:
        nbrs = [[] for _ in range(n)]
        for i, j, _ in edges:
            if i < n and j < n:
                nbrs[i].append(j)
                nbrs[j].append(i)
        return nbrs

    def _force_layout(
        self,
        coords: np.ndarray,
        edges: list[tuple[int, int, float]],
        force: float = 1.0,
        iterations: int = 140,
    ) -> np.ndarray:
        n = len(coords)
        if n == 0:
            return coords
        pos = coords.astype(np.float32)
        area = 1.0
        base_k = math.sqrt(area / max(n, 1))
        k = base_k * max(force, 0.1)
        step = 0.08

        for _ in range(iterations):
            disp = np.zeros_like(pos)
            for i in range(n):
                for j in range(i + 1, n):
                    delta = pos[i] - pos[j]
                    dist = math.hypot(delta[0], delta[1]) + 1e-6
                    rep = (k * k) / dist
                    disp[i] += delta / dist * rep
                    disp[j] -= delta / dist * rep

            for i, j, w in edges:
                delta = pos[i] - pos[j]
                dist = math.hypot(delta[0], delta[1]) + 1e-6
                attr = (dist * dist) / k * (0.6 + w)
                move = delta / dist * attr
                disp[i] -= move
                disp[j] += move

            lengths = np.linalg.norm(disp, axis=1)[:, None] + 1e-9
            pos += (disp / lengths) * step
            pos = np.clip(pos, 0.0, 1.0)

        return pos

    # --- Real-time physics tick ----------------------------------------------------

    def _start_physics(self) -> None:
        self._stop_physics()
        if self.graph_positions is None or self.graph_edges is None:
            return
        self.stable_ticks = 0
        self.sim_start_time = time.time()
        self.sim_job = self.root.after(30, self._tick_physics)

    def _stop_physics(self) -> None:
        if self.sim_job:
            self.root.after_cancel(self.sim_job)
            self.sim_job = None

    def _tick_physics(self) -> None:
        if self.graph_positions is None or self.graph_edges is None:
            return
        if self.is_building:
            self.sim_job = self.root.after(60, self._tick_physics)
            return
        if self.sim_start_time and (time.time() - self.sim_start_time) > SIM_TIME_LIMIT:
            self._stop_physics()
            self.status_var.set(f"Settled (damping={self.damping_var.get():.2f})")
            return

        pos = self.graph_positions
        vel = self.velocities
        if vel is None:
            vel = np.zeros_like(pos)
        n = len(pos)
        if n == 0:
            return
        # Avoid heavy O(n^2) on very large sets
        if n > 500:
            self.sim_job = self.root.after(120, self._tick_physics)
            return

        force = self.cluster_force_var.get()
        k = math.sqrt(1.0 / max(n, 1)) * max(force * 0.5, 0.06)
        disp = np.zeros_like(pos)

        # repulsion (limit to local neighborhood when large)
        if n <= 80 or not self.neighbor_lists:
            for i in range(n):
                for j in range(i + 1, n):
                    delta = pos[i] - pos[j]
                    dist = math.hypot(delta[0], delta[1]) + 1e-6
                    rep = (k * k) / dist
                    push = delta / dist * rep
                    disp[i] += push
                    disp[j] -= push
        else:
            for i in range(n):
                for j in self.neighbor_lists[i]:
                    if j <= i:
                        continue
                    delta = pos[i] - pos[j]
                    dist = math.hypot(delta[0], delta[1]) + 1e-6
                    rep = (k * k) / dist
                    push = delta / dist * rep
                    disp[i] += push
                    disp[j] -= push

        # attraction
        for i, j, w in self.graph_edges:
            if i >= n or j >= n:
                continue
            delta = pos[i] - pos[j]
            dist = math.hypot(delta[0], delta[1]) + 1e-6
            attr = (dist * dist) / k * (0.45 + 0.6 * w)
            pull = delta / dist * attr
            disp[i] -= pull
            disp[j] += pull

        # integrate
        step = 0.04
        damping = float(self.damping_var.get())
        vel = damping * vel + step * np.clip(disp, -0.18, 0.18)
        pos = pos + vel
        pos = np.clip(pos, 0.0, 1.0)

        self.graph_positions = pos
        self.velocities = vel
        if np.max(np.linalg.norm(vel, axis=1, keepdims=False)) < VEL_THRESHOLD:
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        # draw every other tick to stay smooth
        self._draw_toggle = not self._draw_toggle
        if self._draw_toggle:
            self.draw_graph()
        if self.stable_ticks >= STABLE_TICKS_REQUIRED:
            self.last_stable_damping = damping
            self._stop_physics()
            self.status_var.set(f"Settled (damping={damping:.2f})")
        else:
            self.sim_job = self.root.after(60, self._tick_physics)

    # --- Graph UI -------------------------------------------------------------------

    def _schedule_relayout(self) -> None:
        if self._relayout_after_id:
            self.root.after_cancel(self._relayout_after_id)
        self._relayout_after_id = self.root.after(260, self.relayout_graph)

    def _on_damping_change(self) -> None:
        # tweak damping live; restart gentle physics to settle
        self.stable_ticks = 0
        self._start_physics()

    def relayout_graph(self) -> None:
        if self.graph_coords is None or self.graph_edges is None:
            return

        coords = self.graph_coords.copy()
        edges = self.graph_edges
        force = self.cluster_force_var.get()

        def _worker() -> None:
            positions = self._force_layout(coords, edges, force=force, iterations=60)

            def _apply() -> None:
                self.graph_positions = positions
                self.velocities = np.zeros_like(positions)
                self.neighbor_lists = self._build_neighbor_lists(len(positions), edges)
                self.draw_graph()
                self._start_physics()

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def draw_graph(self, highlight_path: Path | None = None) -> None:
        if self.is_building:
            return
        if self.graph_positions is None or not len(self.graph_node_paths):
            self.graph_canvas.delete("all")
            self.graph_canvas.create_text(
                self.graph_canvas.winfo_width() / 2,
                self.graph_canvas.winfo_height() / 2,
                text="Load a folder to see the map",
                fill="#888888",
            )
            return

        width = max(self.graph_canvas.winfo_width(), 200)
        height = max(self.graph_canvas.winfo_height(), 200)
        pad = 32

        self.graph_canvas.delete("all")
        self.graph_canvas_ids.clear()
        self.screen_positions = []

        def to_canvas(pt: np.ndarray) -> tuple[float, float]:
            return pad + pt[0] * (width - 2 * pad), pad + pt[1] * (height - 2 * pad)

        for i, j, w in self.graph_edges:
            if i >= len(self.graph_positions) or j >= len(self.graph_positions):
                continue
            x1, y1 = to_canvas(self.graph_positions[i])
            x2, y2 = to_canvas(self.graph_positions[j])
            shade = max(60, min(200, int(80 + 120 * w)))
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            self.graph_canvas.create_line(x1, y1, x2, y2, fill=color, width=1)

        for idx, path in enumerate(self.graph_node_paths):
            if idx >= len(self.graph_positions):
                continue
            x, y = to_canvas(self.graph_positions[idx])
            r = 8
            is_highlight = highlight_path and path == highlight_path
            fill = "#6ae3ff" if not is_highlight else "#ffde7a"
            outline = "#0d1117" if not is_highlight else "#ffae00"
            node_id = self.graph_canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
            self.graph_canvas_ids[node_id] = path
            self.screen_positions.append((x, y))

    def on_graph_click(self, event: tk.Event) -> None:
        if not self.graph_positions or not self.graph_canvas_ids:
            return
        click_x, click_y = event.x, event.y
        best_idx = None
        best_dist = 1e12
        radius_px = 14  # click hit radius
        for idx, (x, y) in enumerate(self.screen_positions):
            dist = (x - click_x) ** 2 + (y - click_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None or best_dist > radius_px * radius_px:
            return

        best_path = self.graph_node_paths[best_idx]

        self.current_path = best_path
        self.load_audio(best_path)
        self.draw_graph(highlight_path=best_path)

    # --- Audio ----------------------------------------------------------------------

    def load_audio(self, path: Path) -> None:
        try:
            audio, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not open file:\n{path}\n\n{e}")
            return

        self.audio = audio
        self.sample_rate = sr
        frames, channels = audio.shape
        duration = frames / sr
        peak = float(np.max(np.abs(audio))) if frames else 0.0
        self.info_var.set(f"{path.name} — {duration:.2f}s, {channels}ch, peak {peak:.3f}")
        self.play_current()

    def play_current(self) -> None:
        if self.audio is None or self.sample_rate is None:
            return
        audio = self.audio
        sr = self.sample_rate
        self.stop_audio()

        def _worker() -> None:
            try:
                sd.check_output_settings(samplerate=sr, channels=audio.shape[1])
                self.is_playing = True
                sd.play(audio, sr, blocking=True)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Playback Error", str(e)))
            finally:
                self.is_playing = False

        self.play_thread = threading.Thread(target=_worker, daemon=True)
        self.play_thread.start()

    def stop_audio(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass
        self.is_playing = False

    def on_close(self) -> None:
        self.stop_audio()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SampleExplorerApp(root)
    root.mainloop()
