use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::hash::{Hash, Hasher};

use anyhow::{anyhow, Result};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Vec2, ViewportBuilder};
use eframe::{App, Frame};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::prelude::*;
use rodio::{OutputStream, Sink, Source};
use rustfft::{num_complex::Complex, FftPlanner};

static AUDIO_EXT: Lazy<Vec<&str>> = Lazy::new(|| vec!["wav", "aiff", "aif", "flac", "ogg", "mp3"]);
static AUDIO_CACHE: Lazy<RwLock<HashMap<PathBuf, Vec<f32>>>> = Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Clone)]
struct SampleNode {
    path: PathBuf,
    rms: f32,
    centroid: f32,
    zcr: f32,
    vec: Vec<f32>,
    pos: Vec2,
    vel: Vec2,
    color: Color32,
}

struct AudioPlayer {
    _stream: OutputStream,
    sink: Sink,
}

impl AudioPlayer {
    fn new() -> Result<Self> {
        let (_stream, handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&handle)?;
        Ok(Self { _stream, sink })
    }

    fn play(&self, samples: &[f32], sample_rate: u32) -> Result<()> {
        self.sink.stop();
        let src = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples.to_vec());
        self.sink.append(src);
        self.sink.play();
        Ok(())
    }
}

struct ExplorerApp {
    folder: Option<PathBuf>,
    nodes: Vec<SampleNode>,
    player: Option<AudioPlayer>,
    cluster_force: f32,
    damping: f32,
    status: String,
    last_layout_ms: u128,
    sim_active: bool,
    sim_start: Option<Instant>,
    neighbor_lists: Vec<Vec<usize>>,
    settle_frames: usize,
}

impl ExplorerApp {
    fn new() -> Self {
        Self {
            folder: None,
            nodes: Vec::new(),
            player: AudioPlayer::new().ok(),
            cluster_force: 0.7,
            damping: 0.992,
            status: "Open a folder".into(),
            last_layout_ms: 0,
            sim_active: false,
            sim_start: None,
            neighbor_lists: Vec::new(),
            settle_frames: 0,
        }
    }

    fn choose_folder(&mut self) {
        if let Some(p) = rfd::FileDialog::new().set_directory(".").pick_folder() {
            self.folder = Some(p.clone());
            self.status = "Scanning…".into();
            self.scan_folder();
        }
    }

    fn scan_folder(&mut self) {
        let Some(folder) = &self.folder else { return };
        let files: Vec<PathBuf> = walkdir::WalkDir::new(folder)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| AUDIO_EXT.iter().any(|ext| s.eq_ignore_ascii_case(ext)))
                    .unwrap_or(false)
            })
            .map(|e| e.into_path())
            .collect();

        let start = Instant::now();
        let nodes: Vec<SampleNode> = files
            .par_iter()
            .filter_map(|p| feature_node(p).ok())
            .collect();
        let mut nodes = nodes;
        // Normalize vectors
        for n in &mut nodes {
            let norm: f32 = n.vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-5);
            for v in &mut n.vec {
                *v /= norm;
            }
        }
        let colors = assign_colors(folder, &nodes);
        for n in nodes.iter_mut() {
            if let Some(c) = colors.get(&n.path) {
                n.color = *c;
            }
        }
        layout_nodes(&mut nodes, self.cluster_force, self.damping);
        self.neighbor_lists = build_neighbors(&nodes, 8);
        for n in nodes.iter_mut() {
            n.vel = Vec2::ZERO;
        }
        self.sim_active = true;
        self.sim_start = Some(Instant::now());
        self.last_layout_ms = start.elapsed().as_millis();
        self.status = format!("Loaded {} samples ({} ms)", nodes.len(), self.last_layout_ms);
        self.nodes = nodes;
    }

    fn click_play(&mut self, path: &Path) {
        let Ok(data) = load_audio_mono(path) else {
            self.status = format!("Failed to play {:?}", path.file_name().unwrap_or_default());
            return;
        };
        if let Some(player) = &self.player {
            let _ = player.play(&data, 44_100); // we resample-ish below
        }
    }

    fn restart_sim(&mut self) {
        for n in &mut self.nodes {
            n.vel = Vec2::ZERO;
        }
        self.sim_active = true;
        self.sim_start = Some(Instant::now());
        self.settle_frames = 0;
    }
}

impl App for ExplorerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open Folder").clicked() {
                    self.choose_folder();
                }
                ui.label(format!("Force: {:.2}", self.cluster_force));
                if ui
                    .add(egui::Slider::new(&mut self.cluster_force, 0.2..=2.0).show_value(false))
                    .changed()
                {
                    layout_nodes(&mut self.nodes, self.cluster_force, self.damping);
                    self.neighbor_lists = build_neighbors(&self.nodes, 8);
                    self.restart_sim();
                }
                ui.label(format!("Damping: {:.3}", self.damping));
                if ui
                    .add(egui::Slider::new(&mut self.damping, 0.93..=0.9995).show_value(false))
                    .changed()
                {
                    layout_nodes(&mut self.nodes, self.cluster_force, self.damping);
                    self.restart_sim();
                }
                ui.label(&self.status);
                ui.separator();
                ui.label(format!("Layout: {} ms", self.last_layout_ms));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (response, painter) = ui.allocate_painter(avail, Sense::click());
            let rect = response.rect;
            draw_graph(&self.nodes, &painter, rect);

            if response.clicked() || response.double_clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    if let Some(idx) = pick_node(&self.nodes, pos, rect) {
                        let path = self.nodes[idx].path.clone();
                        self.click_play(&path);
                    }
                }
            }
        });

        if self.sim_active {
            let dt = 1.0 / 60.0;
            step_sim(
                &mut self.nodes,
                &self.neighbor_lists,
                self.cluster_force,
                self.damping,
                dt,
            );
            let speed = max_speed(&self.nodes);
            if speed < 0.0008 {
                self.settle_frames += 1;
            } else {
                self.settle_frames = 0;
            }
            if self.settle_frames >= 30 {
                self.sim_active = false;
                self.status = format!("Settled (damping {:.3})", self.damping);
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            }
        }
    }
}

fn draw_graph(nodes: &[SampleNode], painter: &egui::Painter, rect: Rect) {
    for (i, a) in nodes.iter().enumerate() {
        for (_j, b) in nodes.iter().enumerate().skip(i + 1) {
            let pa = lerp_pos(rect, a.pos);
            let pb = lerp_pos(rect, b.pos);
            let d2 = pa.distance_sq(pb);
            if d2 < 90.0 * 90.0 {
                let gray = (220.0 - (d2.sqrt() / 90.0) * 120.0).clamp(80.0, 220.0) as u8;
                painter.line_segment([pa, pb], egui::Stroke::new(0.7, Color32::from_gray(gray)));
            }
        }
    }

    for node in nodes {
        let p = lerp_pos(rect, node.pos);
        painter.circle_filled(p, 7.0, node.color);
    }
}

fn pick_node(nodes: &[SampleNode], mouse: Pos2, rect: Rect) -> Option<usize> {
    let mut best = None;
    let mut best_d = 18.0f32 * 18.0;
    for (i, n) in nodes.iter().enumerate() {
        let p = lerp_pos(rect, n.pos);
        let d = p.distance_sq(mouse);
        if d < best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

fn lerp_pos(rect: Rect, p: Vec2) -> Pos2 {
    Pos2::new(
        rect.left() + p.x * rect.width(),
        rect.top() + p.y * rect.height(),
    )
}

fn feature_node(path: &Path) -> Result<SampleNode> {
    let samples = load_audio_mono(path)?;
    if samples.is_empty() {
        return Err(anyhow!("empty"));
    }
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
    let zcr = samples
        .windows(2)
        .filter(|w| (w[0].is_sign_positive()) != (w[1].is_sign_positive()))
        .count() as f32
        / samples.len() as f32;
    let (centroid, rolloff, flatness) = spectral_shape(&samples, 44_100);

    let mut vec = vec![rms, zcr, centroid, rolloff, flatness];
    vec.extend(band_energy(&samples, 44_100, 16));

    Ok(SampleNode {
        path: path.to_path_buf(),
        rms,
        centroid,
        zcr,
        vec,
        pos: Vec2::new(rand_unit(), rand_unit()),
        vel: Vec2::ZERO,
        color: Color32::from_rgb(110, 227, 255),
    })
}

fn rand_unit() -> f32 {
    use rand::Rng;
    static RNG: Lazy<parking_lot::Mutex<rand::rngs::StdRng>> =
        Lazy::new(|| parking_lot::Mutex::new(rand::SeedableRng::seed_from_u64(42)));
    let mut rng = RNG.lock();
    rng.gen::<f32>()
}

fn band_energy(samples: &[f32], sr: u32, bands: usize) -> Vec<f32> {
    let n = samples.len().next_power_of_two().min(4096).max(256);
    let mut buf: Vec<Complex<f32>> = samples
        .iter()
        .take(n)
        .map(|&s| Complex::new(s, 0.0))
        .collect();
    buf.resize(n, Complex::new(0.0, 0.0));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let mags: Vec<f32> = buf.iter().map(|c| c.norm_sqr()).collect();
    let freqs: Vec<f32> = (0..n).map(|i| i as f32 * sr as f32 / n as f32).collect();
    let edges = logspace(30.0, sr as f32 / 2.0, bands + 1);
    let mut out = Vec::with_capacity(bands);
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut sum = 0.0;
        let mut count = 0;
        for (f, m) in freqs.iter().zip(mags.iter()) {
            if *f >= lo && *f < hi {
                sum += *m;
                count += 1;
            }
        }
        out.push(if count == 0 { 0.0 } else { sum / count as f32 });
    }
    out
}

fn assign_colors(folder: &Path, nodes: &[SampleNode]) -> HashMap<PathBuf, Color32> {
    let palette = [
        Color32::from_rgb(110, 227, 255),
        Color32::from_rgb(255, 204, 112),
        Color32::from_rgb(173, 214, 255),
        Color32::from_rgb(255, 158, 177),
        Color32::from_rgb(168, 255, 185),
        Color32::from_rgb(255, 191, 143),
        Color32::from_rgb(192, 186, 255),
        Color32::from_rgb(255, 255, 153),
    ];
    let mut group_color: HashMap<String, Color32> = HashMap::new();
    let mut out = HashMap::new();
    for n in nodes {
        let group = n
            .path
            .strip_prefix(folder)
            .ok()
            .and_then(|p| p.parent())
            .and_then(|p| p.components().next())
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "root".into());
        let color = *group_color.entry(group.clone()).or_insert_with(|| {
            let mut hasher = DefaultHasher::new();
            group.hash(&mut hasher);
            let idx = (hasher.finish() % palette.len() as u64) as usize;
            palette[idx]
        });
        out.insert(n.path.clone(), color);
    }
    out
}

fn spectral_shape(samples: &[f32], sr: u32) -> (f32, f32, f32) {
    let n = samples.len().next_power_of_two().min(4096).max(256);
    let mut buf: Vec<Complex<f32>> = samples
        .iter()
        .take(n)
        .map(|&s| Complex::new(s, 0.0))
        .collect();
    buf.resize(n, Complex::new(0.0, 0.0));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let mags: Vec<f32> = buf.iter().map(|c| c.norm_sqr().max(1e-12)).collect();
    let freqs: Vec<f32> = (0..n).map(|i| i as f32 * sr as f32 / n as f32).collect();
    let num: f32 = freqs.iter().zip(mags.iter()).map(|(f, m)| f * m).sum();
    let den: f32 = mags.iter().sum::<f32>().max(1e-6);
    let centroid = num / den;
    let cumulative: Vec<f32> = mags
        .iter()
        .scan(0.0, |acc, v| {
            *acc += *v;
            Some(*acc)
        })
        .collect();
    let target = den * 0.85;
    let mut rolloff = sr as f32 / 2.0;
    for (i, c) in cumulative.iter().enumerate() {
        if *c >= target {
            rolloff = freqs[i];
            break;
        }
    }
    let geo = mags.iter().map(|m| m.ln()).sum::<f32>().exp().powf(1.0 / mags.len() as f32);
    let flatness = (geo / (den / mags.len() as f32)).clamp(0.0, 1.0);
    (centroid, rolloff, flatness)
}

fn logspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    let log_start = start.ln();
    let log_end = end.ln();
    (0..n)
        .map(|i| {
            let t = i as f32 / (n as f32 - 1.0);
            (log_start + t * (log_end - log_start)).exp()
        })
        .collect()
}

fn layout_nodes(nodes: &mut [SampleNode], force: f32, damping: f32) {
    if nodes.is_empty() {
        return;
    }
    // Start from random positions
    for n in nodes.iter_mut() {
        n.pos = Vec2::new(rand_unit(), rand_unit());
    }

    let k = (1.0 / nodes.len() as f32).sqrt();
    let iter = 120;
    let alpha0 = 0.05 * force;
    for t in 0..iter {
        let alpha = alpha0 * (1.0 - t as f32 / iter as f32);
        for i in 0..nodes.len() {
            let mut disp = Vec2::ZERO;
            for j in (i + 1)..nodes.len() {
                let delta = nodes[i].pos - nodes[j].pos;
                let dist = (delta.length() + 1e-3).max(1e-3);
                let rep = (k * k) / dist;
                let push = delta / dist * rep;
                disp += push;
                nodes[j].pos -= push;
            }
            // attraction via cosine similarity
            let mut total_attr = Vec2::ZERO;
            for j in 0..nodes.len() {
                if i == j {
                    continue;
                }
                let sim = cosine(&nodes[i].vec, &nodes[j].vec).max(0.0);
                if sim <= 0.0 {
                    continue;
                }
                let delta = nodes[i].pos - nodes[j].pos;
                let dist = (delta.length() + 1e-3).max(1e-3);
                let attr = (dist * dist) / k * sim * force;
                total_attr -= delta / dist * attr;
            }
            nodes[i].pos += (disp + total_attr) * alpha * damping;
            nodes[i].pos.x = nodes[i].pos.x.clamp(0.0, 1.0);
            nodes[i].pos.y = nodes[i].pos.y.clamp(0.0, 1.0);
        }
    }
}

fn step_sim(
    nodes: &mut [SampleNode],
    neighbors: &[Vec<usize>],
    force: f32,
    damping: f32,
    _dt: f32,
) {
    let n = nodes.len();
    if n == 0 {
        return;
    }
    let k = (1.0 / n as f32).sqrt() * (0.5 + force * 0.4).clamp(0.35, 0.8);
    for i in 0..n {
        let mut disp = Vec2::ZERO;
        // limited repulsion: random sample of others and neighbors
        for j in neighbors.get(i).into_iter().flatten() {
            let delta = nodes[i].pos - nodes[*j].pos;
            let dist = (delta.length() + 1e-3).max(1e-3);
            let rep = (k * k) / dist;
            disp += delta / dist * rep;
        }
        // light global jitter repulsion
        // very light global jitter repulsion to prevent overlap
        if i % 7 == 0 {
            for j in (i + 1)..n {
                if j % 17 != 0 {
                    continue;
                }
                let delta = nodes[i].pos - nodes[j].pos;
                let dist = (delta.length() + 1e-3).max(1e-3);
                let rep = (k * k) / dist * 0.12;
                disp += delta / dist * rep;
            }
        }
        let mut attr = Vec2::ZERO;
        for j in neighbors.get(i).into_iter().flatten() {
            let sim = cosine(&nodes[i].vec, &nodes[*j].vec).max(0.0);
            if sim <= 0.0 {
                continue;
            }
            let delta = nodes[i].pos - nodes[*j].pos;
            let dist = (delta.length() + 1e-3).max(1e-3);
            let a = (dist * dist) / k * sim * force;
            attr -= delta / dist * a;
        }
        let acc = disp + attr;
        let max_len = 0.04;
        let len = acc.length();
        let limited = if len > max_len { acc / len * max_len } else { acc };
        nodes[i].vel = nodes[i].vel * damping + limited;
        nodes[i].pos += nodes[i].vel;
        nodes[i].pos.x = nodes[i].pos.x.clamp(0.0, 1.0);
        nodes[i].pos.y = nodes[i].pos.y.clamp(0.0, 1.0);
    }
}

fn build_neighbors(nodes: &[SampleNode], k: usize) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let mut out = vec![Vec::new(); n];
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&nodes[i].vec, &nodes[j].vec)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (j, _) in sims.into_iter().take(k) {
            out[i].push(j);
        }
    }
    out
}

fn max_speed(nodes: &[SampleNode]) -> f32 {
    nodes
        .iter()
        .map(|n| n.vel.length())
        .fold(0.0, |a, b| a.max(b))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = (a.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
    let nb = (b.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
    dot / (na * nb)
}

fn load_audio_mono(path: &Path) -> Result<Vec<f32>> {
    // cache to avoid re-decoding
    if let Some(cached) = AUDIO_CACHE.read().get(path).cloned() {
        return Ok(cached);
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let mut data = if ext == "wav" {
        read_wav(path)?
    } else {
        let file = File::open(path)?;
        let src = rodio::Decoder::new(BufReader::new(file))?;
        src.convert_samples::<f32>()
            .take(44_100 * 12) // 12 seconds max
            .collect::<Vec<f32>>()
    };

    // normalize & downmix
    if data.is_empty() {
        return Err(anyhow!("empty"));
    }
    let peak = data.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    for v in &mut data {
        *v /= peak;
    }

    let mut cache = AUDIO_CACHE.write();
    cache.insert(path.to_path_buf(), data.clone());
    Ok(data)
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / i32::MAX as f32)
        .collect();
    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::with_capacity(samples.len() / channels);
    for chunk in samples.chunks(channels) {
        let avg = chunk.iter().copied().sum::<f32>() / channels as f32;
        mono.push(avg);
    }
    // simple resample to 44.1k if needed
    if sr != 44_100 {
        let ratio = 44_100.0 / sr as f32;
        let new_len = (mono.len() as f32 * ratio) as usize;
        let mut res = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f32 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = src_pos - idx as f32;
            let s0 = *mono.get(idx).unwrap_or(&0.0);
            let s1 = *mono.get(idx + 1).unwrap_or(&s0);
            res.push(s0 * (1.0 - frac) + s1 * frac);
        }
        Ok(res)
    } else {
        Ok(mono)
    }
}

fn main() -> Result<()> {
    let viewport = ViewportBuilder::default().with_inner_size([1200.0, 800.0]);
    let options = eframe::NativeOptions {
        viewport: viewport.into(),
        ..Default::default()
    };
    eframe::run_native(
        "Sample Explorer (Rust)",
        options,
        Box::new(|_cc| Box::new(ExplorerApp::new())),
    )
    .map_err(|e| anyhow!(e.to_string()))
}
