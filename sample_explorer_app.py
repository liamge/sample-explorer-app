import threading
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


class SampleExplorerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sample Explorer")
        self.root.geometry("1180x720")

        self.current_folder: Path | None = None
        self.sample_paths: list[Path] = []
        self.visible_paths: list[Path] = []
        self.current_path: Path | None = None

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.play_thread: threading.Thread | None = None
        self.is_playing = False

        self.trim_start_var = tk.DoubleVar(value=0.0)
        self.trim_end_var = tk.DoubleVar(value=1.0)
        self.fade_in_var = tk.DoubleVar(value=0.01)
        self.fade_out_var = tk.DoubleVar(value=0.03)
        self.normalize_var = tk.BooleanVar(value=True)
        self.reverse_var = tk.BooleanVar(value=False)
        self.mono_var = tk.BooleanVar(value=False)
        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a folder to begin.")

        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self.refresh_file_list())

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        topbar = ttk.Frame(outer)
        topbar.pack(fill="x", pady=(0, 10))

        ttk.Button(topbar, text="Open Folder", command=self.choose_folder).pack(side="left")
        ttk.Button(topbar, text="Rescan", command=self.scan_folder).pack(side="left", padx=(8, 0))
        ttk.Label(topbar, text="Search:").pack(side="left", padx=(18, 6))
        ttk.Entry(topbar, textvariable=self.search_var, width=28).pack(side="left")
        ttk.Label(topbar, textvariable=self.status_var).pack(side="right")

        content = ttk.Panedwindow(outer, orient="horizontal")
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, padding=8)
        right = ttk.Frame(content, padding=8)
        content.add(left, weight=1)
        content.add(right, weight=3)

        ttk.Label(left, text="Samples").pack(anchor="w")

        self.file_list = tk.Listbox(left, activestyle="dotbox")
        self.file_list.pack(fill="both", expand=True, pady=(6, 6))
        self.file_list.bind("<<ListboxSelect>>", self.on_select_file)
        self.file_list.bind("<Double-1>", lambda e: self.play_current())

        file_btns = ttk.Frame(left)
        file_btns.pack(fill="x")
        ttk.Button(file_btns, text="Play", command=self.play_current).pack(side="left", fill="x", expand=True)
        ttk.Button(file_btns, text="Stop", command=self.stop_audio).pack(side="left", fill="x", expand=True, padx=6)

        info_frame = ttk.LabelFrame(right, text="Preview")
        info_frame.pack(fill="both", expand=False)

        self.file_info = tk.Text(info_frame, height=5, wrap="word")
        self.file_info.pack(fill="x", padx=8, pady=8)
        self.file_info.configure(state="disabled")

        self.wave_canvas = tk.Canvas(right, height=220, bg="#111111", highlightthickness=1)
        self.wave_canvas.pack(fill="x", pady=(10, 12))

        controls = ttk.LabelFrame(right, text="Create / Process")
        controls.pack(fill="x")

        grid = ttk.Frame(controls, padding=10)
        grid.pack(fill="x")

        ttk.Label(grid, text="Trim start").grid(row=0, column=0, sticky="w")
        self.trim_start = ttk.Scale(
            grid,
            from_=0.0,
            to=1.0,
            variable=self.trim_start_var,
            command=lambda _: self.redraw_waveform(),
        )
        self.trim_start.grid(row=0, column=1, sticky="ew", padx=8)
        self.trim_start_label = ttk.Label(grid, text="0%")
        self.trim_start_label.grid(row=0, column=2, sticky="e")

        ttk.Label(grid, text="Trim end").grid(row=1, column=0, sticky="w")
        self.trim_end = ttk.Scale(
            grid,
            from_=0.0,
            to=1.0,
            variable=self.trim_end_var,
            command=lambda _: self.redraw_waveform(),
        )
        self.trim_end.grid(row=1, column=1, sticky="ew", padx=8)
        self.trim_end_label = ttk.Label(grid, text="100%")
        self.trim_end_label.grid(row=1, column=2, sticky="e")

        ttk.Label(grid, text="Fade in (sec)").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0.0, to=2.0, increment=0.01, textvariable=self.fade_in_var, width=10).grid(
            row=2, column=1, sticky="w", padx=8
        )

        ttk.Label(grid, text="Fade out (sec)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0.0, to=2.0, increment=0.01, textvariable=self.fade_out_var, width=10).grid(
            row=3, column=1, sticky="w", padx=8
        )

        ttk.Checkbutton(grid, text="Normalize", variable=self.normalize_var).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(grid, text="Reverse", variable=self.reverse_var).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(grid, text="Convert to mono", variable=self.mono_var).grid(row=4, column=2, sticky="w")

        grid.columnconfigure(1, weight=1)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(12, 0))
        ttk.Button(btns, text="Preview Processed", command=self.preview_processed).pack(side="left")
        ttk.Button(btns, text="Export Processed Sample", command=self.export_processed).pack(side="left", padx=8)
        ttk.Button(btns, text="Duplicate as One-Shot", command=self.export_one_shot).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        self.refresh_file_list()
        self.status_var.set(f"Found {len(self.sample_paths)} files")

    def refresh_file_list(self) -> None:
        query = self.search_var.get().strip().lower()
        self.file_list.delete(0, tk.END)
        self.visible_paths = []
        for path in self.sample_paths:
            rel = str(path.relative_to(self.current_folder)) if self.current_folder else path.name
            if query and query not in rel.lower():
                continue
            self.visible_paths.append(path)
            self.file_list.insert(tk.END, rel)

    def on_select_file(self, _event=None) -> None:
        selection = self.file_list.curselection()
        if not selection:
            return
        self.current_path = self.visible_paths[selection[0]]
        self.load_audio(self.current_path)

    def load_audio(self, path: Path) -> None:
        try:
            audio, sr = sf.read(path, always_2d=True, dtype="float32")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not open file:\n{path}\n\n{e}")
            return

        self.audio = audio
        self.sample_rate = sr
        self.trim_start_var.set(0.0)
        self.trim_end_var.set(1.0)
        self.update_file_info(path, audio, sr)
        self.redraw_waveform()

    def update_file_info(self, path: Path, audio: np.ndarray, sr: int) -> None:
        frames, channels = audio.shape
        duration = frames / sr
        peak = float(np.max(np.abs(audio))) if frames else 0.0
        text = (
            f"File: {path.name}\n"
            f"Path: {path}\n"
            f"Sample rate: {sr} Hz\n"
            f"Channels: {channels}\n"
            f"Duration: {duration:.2f} sec\n"
            f"Peak: {peak:.3f}"
        )
        self.file_info.configure(state="normal")
        self.file_info.delete("1.0", tk.END)
        self.file_info.insert("1.0", text)
        self.file_info.configure(state="disabled")

    def redraw_waveform(self) -> None:
        self.wave_canvas.delete("all")
        if self.audio is None:
            return

        width = max(self.wave_canvas.winfo_width(), 400)
        height = max(self.wave_canvas.winfo_height(), 220)
        center_y = height / 2

        mono = self.audio.mean(axis=1)
        if len(mono) == 0:
            return

        step = max(1, len(mono) // width)
        reduced = mono[::step][:width]
        peak = np.max(np.abs(reduced)) or 1.0
        scaled = reduced / peak

        points: list[float] = []
        for i, v in enumerate(scaled):
            y = center_y - (v * (height * 0.42))
            points.extend([i, y])

        self.wave_canvas.create_line(points, fill="#77ddee", width=1)
        self.wave_canvas.create_line(0, center_y, width, center_y, fill="#444444")

        start_x = self.trim_start_var.get() * width
        end_x = self.trim_end_var.get() * width
        if end_x < start_x:
            start_x, end_x = end_x, start_x

        self.wave_canvas.create_rectangle(0, 0, start_x, height, fill="#000000", stipple="gray25", outline="")
        self.wave_canvas.create_rectangle(end_x, 0, width, height, fill="#000000", stipple="gray25", outline="")
        self.wave_canvas.create_line(start_x, 0, start_x, height, fill="#ffcc66", width=2)
        self.wave_canvas.create_line(end_x, 0, end_x, height, fill="#ffcc66", width=2)

        self.trim_start_label.configure(text=f"{self.trim_start_var.get() * 100:.0f}%")
        self.trim_end_label.configure(text=f"{self.trim_end_var.get() * 100:.0f}%")

    def get_processed_audio(self) -> tuple[np.ndarray, int] | None:
        if self.audio is None or self.sample_rate is None:
            return None

        audio = self.audio.copy()
        sr = self.sample_rate

        total_frames = len(audio)
        start = int(min(self.trim_start_var.get(), self.trim_end_var.get()) * total_frames)
        end = int(max(self.trim_start_var.get(), self.trim_end_var.get()) * total_frames)
        end = max(end, start + 1)
        audio = audio[start:end]

        if self.mono_var.get() and audio.ndim == 2:
            audio = audio.mean(axis=1, keepdims=True)

        if self.reverse_var.get():
            audio = audio[::-1]

        fade_in_samples = min(int(self.fade_in_var.get() * sr), len(audio))
        fade_out_samples = min(int(self.fade_out_var.get() * sr), len(audio))

        if fade_in_samples > 0:
            fade = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)[:, None]
            audio[:fade_in_samples] *= fade

        if fade_out_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)[:, None]
            audio[-fade_out_samples:] *= fade

        if self.normalize_var.get():
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.98

        return audio.astype(np.float32), sr

    def play_array(self, audio: np.ndarray, sr: int) -> None:
        self.stop_audio()

        def _worker() -> None:
            try:
                self.is_playing = True
                sd.play(audio, sr)
                sd.wait()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Playback Error", str(e)))
            finally:
                self.is_playing = False

        self.play_thread = threading.Thread(target=_worker, daemon=True)
        self.play_thread.start()

    def play_current(self) -> None:
        if self.audio is None or self.sample_rate is None:
            return
        self.play_array(self.audio, self.sample_rate)

    def preview_processed(self) -> None:
        processed = self.get_processed_audio()
        if processed is None:
            return
        audio, sr = processed
        self.play_array(audio, sr)

    def stop_audio(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass
        self.is_playing = False

    def export_processed(self) -> None:
        processed = self.get_processed_audio()
        if processed is None or self.current_path is None:
            return
        audio, sr = processed

        initial_name = self.current_path.stem + "_processed.wav"
        out_path = filedialog.asksaveasfilename(
            title="Export processed sample",
            defaultextension=".wav",
            initialfile=initial_name,
            filetypes=[("WAV file", "*.wav"), ("AIFF file", "*.aiff")],
        )
        if not out_path:
            return

        try:
            sf.write(out_path, audio, sr)
            self.status_var.set(f"Saved {Path(out_path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_one_shot(self) -> None:
        processed = self.get_processed_audio()
        if processed is None or self.current_path is None:
            return
        audio, sr = processed

        target_len = int(1.0 * sr)
        if len(audio) < target_len:
            pad = np.zeros((target_len - len(audio), audio.shape[1]), dtype=np.float32)
            audio = np.vstack([audio, pad])

        out_path = filedialog.asksaveasfilename(
            title="Export one-shot sample",
            defaultextension=".wav",
            initialfile=self.current_path.stem + "_oneshot.wav",
            filetypes=[("WAV file", "*.wav")],
        )
        if not out_path:
            return

        try:
            sf.write(out_path, audio, sr)
            self.status_var.set(f"Saved {Path(out_path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def on_close(self) -> None:
        self.stop_audio()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SampleExplorerApp(root)

    def _delayed_redraw() -> None:
        app.redraw_waveform()
        root.after(300, _delayed_redraw)

    root.after(300, _delayed_redraw)
    root.mainloop()
