use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
            cluster_force: 0.95,
            damping: 0.90,
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
        let mut nodes: Vec<SampleNode> = files.par_iter().filter_map(|p| feature_node(p).ok()).collect();

        standardize_features(&mut nodes);

        let colors = assign_colors(folder, &nodes);
        for n in nodes.iter_mut() {
            if let Some(c) = colors.get(&n.path) {
                n.color = *c;
            }
        }

        let anchor = build_anchor_positions(&nodes);
        for (n, p) in nodes.iter_mut().zip(anchor.into_iter()) {
            n.pos = p;
            n.vel = Vec2::ZERO;
        }

        self.neighbor_lists = build_neighbors(&nodes, 10);
        self.sim_active = true;
        self.sim_start = Some(Instant::now());
        self.settle_frames = 0;
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
            let _ = player.play(&data, 44_100);
        }
    }

    fn restart_sim(&mut self) {
        let anchors = build_anchor_positions(&self.nodes);
        for (n, anchor) in self.nodes.iter_mut().zip(anchors.into_iter()) {
            n.pos = anchor;
            n.vel = Vec2::ZERO;
        }
        self.neighbor_lists = build_neighbors(&self.nodes, 10);
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
                ui.label(format!("Cluster: {:.2}", self.cluster_force));
                if ui
                    .add(egui::Slider::new(&mut self.cluster_force, 0.35..=1.75).show_value(false))
                    .changed()
                {
                    self.restart_sim();
                }
                ui.label(format!("Damping: {:.2}", self.damping));
                if ui
                    .add(egui::Slider::new(&mut self.damping, 0.78..=0.97).show_value(false))
                    .changed()
                {
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
            draw_graph(&self.nodes, &self.neighbor_lists, &painter, rect);

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
            if speed < 0.00045 {
                self.settle_frames += 1;
            } else {
                self.settle_frames = 0;
            }
            if self.settle_frames >= 45 {
                self.sim_active = false;
                self.status = format!("Settled (damping {:.2})", self.damping);
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            }
        }
    }
}

fn draw_graph(nodes: &[SampleNode], neighbors: &[Vec<usize>], painter: &egui::Painter, rect: Rect) {
    for (i, a) in nodes.iter().enumerate() {
        for &j in neighbors.get(i).into_iter().flatten() {
            if j <= i {
                continue;
            }
            let b = &nodes[j];
            let pa = lerp_pos(rect, a.pos);
            let pb = lerp_pos(rect, b.pos);
            let sim = cosine(&a.vec, &b.vec).clamp(0.0, 1.0);
            if sim < 0.42 {
                continue;
            }
            let alpha = (35.0 + sim * 85.0) as u8;
            painter.line_segment([pa, pb], egui::Stroke::new(0.9, Color32::from_white_alpha(alpha)));
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
    Pos2::new(rect.left() + p.x * rect.width(), rect.top() + p.y * rect.height())
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
    let (centroid, _, _) = spectral_shape(&samples, 44_100);

    let vec = timbre_features(&samples, 44_100);

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

fn timbre_features(samples: &[f32], sr: u32) -> Vec<f32> {
    let frame = 2048usize;
    let hop = 512usize;
    if samples.len() < frame {
        let mut out = vec![0.0; 34];
        let (centroid, rolloff, flatness) = spectral_shape(samples, sr);
        out[0] = rms(samples);
        out[2] = centroid / 10_000.0;
        out[4] = rolloff / 12_000.0;
        out[6] = flatness;
        return out;
    }

    let window = hann_window(frame);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame);
    let band_edges = [20.0, 80.0, 200.0, 600.0, 2000.0, 6000.0, 12_000.0, sr as f32 / 2.0];

    let mut rms_vals = Vec::new();
    let mut zcr_vals = Vec::new();
    let mut centroid_vals = Vec::new();
    let mut rolloff_vals = Vec::new();
    let mut flatness_vals = Vec::new();
    let mut flux_vals = Vec::new();
    let mut crest_vals = Vec::new();
    let mut attack_vals = Vec::new();
    let mut band_logs: Vec<Vec<f32>> = vec![Vec::new(); band_edges.len() - 1];
    let mut prev_mag: Option<Vec<f32>> = None;

    for start in (0..=samples.len() - frame).step_by(hop) {
        let frame_slice = &samples[start..start + frame];
        rms_vals.push(rms(frame_slice));
        zcr_vals.push(zcr(frame_slice));
        attack_vals.push(onset_strength(frame_slice));

        let mut buf: Vec<Complex<f32>> = frame_slice
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        fft.process(&mut buf);
        let half = frame / 2;
        let mags: Vec<f32> = buf[..half].iter().map(|c| c.norm()).collect();
        let powers: Vec<f32> = mags.iter().map(|m| m * m + 1e-12).collect();
        let freqs: Vec<f32> = (0..half).map(|i| i as f32 * sr as f32 / frame as f32).collect();

        let power_sum: f32 = powers.iter().sum::<f32>().max(1e-6);
        let centroid = freqs.iter().zip(powers.iter()).map(|(f, p)| f * p).sum::<f32>() / power_sum;
        centroid_vals.push(centroid / 10_000.0);

        let target = power_sum * 0.85;
        let mut running = 0.0;
        let mut rolloff = sr as f32 / 2.0;
        for (f, p) in freqs.iter().zip(powers.iter()) {
            running += *p;
            if running >= target {
                rolloff = *f;
                break;
            }
        }
        rolloff_vals.push(rolloff / 12_000.0);

        let mean_power = power_sum / powers.len() as f32;
        let geo_mean = (powers.iter().map(|p| p.ln()).sum::<f32>() / powers.len() as f32).exp();
        flatness_vals.push((geo_mean / mean_power).clamp(0.0, 1.5));

        let max_mag = mags.iter().copied().fold(0.0f32, f32::max);
        let mean_mag = mags.iter().copied().sum::<f32>() / mags.len() as f32;
        crest_vals.push(max_mag / mean_mag.max(1e-6));

        for b in 0..band_edges.len() - 1 {
            let lo = band_edges[b];
            let hi = band_edges[b + 1];
            let mut band_sum = 0.0;
            let mut count = 0;
            for (f, p) in freqs.iter().zip(powers.iter()) {
                if *f >= lo && *f < hi {
                    band_sum += *p;
                    count += 1;
                }
            }
            let avg = if count == 0 { 0.0 } else { band_sum / count as f32 };
            band_logs[b].push((avg + 1e-9).ln());
        }

        if let Some(prev) = &prev_mag {
            let flux = mags
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).max(0.0))
                .sum::<f32>()
                / mags.len() as f32;
            flux_vals.push(flux);
        }
        prev_mag = Some(mags);
    }

    let mut out = Vec::new();
    push_mean_std(&mut out, &rms_vals);
    push_mean_std(&mut out, &zcr_vals);
    push_mean_std(&mut out, &centroid_vals);
    push_mean_std(&mut out, &rolloff_vals);
    push_mean_std(&mut out, &flatness_vals);
    push_mean_std(&mut out, &flux_vals);
    push_mean_std(&mut out, &crest_vals);
    push_mean_std(&mut out, &attack_vals);
    for band in &band_logs {
        push_mean_std(&mut out, band);
    }
    out
}

fn push_mean_std(out: &mut Vec<f32>, vals: &[f32]) {
    if vals.is_empty() {
        out.push(0.0);
        out.push(0.0);
        return;
    }
    let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32;
    out.push(mean);
    out.push(var.sqrt());
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / n as f32).cos())
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|v| v * v).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn zcr(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .filter(|w| (w[0].is_sign_positive()) != (w[1].is_sign_positive()))
        .count() as f32
        / samples.len().max(1) as f32
}

fn onset_strength(samples: &[f32]) -> f32 {
    let mut total = 0.0;
    for w in samples.windows(2) {
        total += (w[1].abs() - w[0].abs()).max(0.0);
    }
    total / samples.len().max(1) as f32
}

fn rand_unit() -> f32 {
    use rand::Rng;
    static RNG: Lazy<parking_lot::Mutex<rand::rngs::StdRng>> =
        Lazy::new(|| parking_lot::Mutex::new(rand::SeedableRng::seed_from_u64(42)));
    let mut rng = RNG.lock();
    rng.gen::<f32>()
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
    let mut buf: Vec<Complex<f32>> = samples.iter().take(n).map(|&s| Complex::new(s, 0.0)).collect();
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
    let geo = (mags.iter().map(|m| m.ln()).sum::<f32>() / mags.len() as f32).exp();
    let flatness = (geo / (den / mags.len() as f32)).clamp(0.0, 1.0);
    (centroid, rolloff, flatness)
}

fn standardize_features(nodes: &mut [SampleNode]) {
    if nodes.is_empty() {
        return;
    }
    let dim = nodes[0].vec.len();
    let n = nodes.len() as f32;
    let mut means = vec![0.0; dim];
    for node in nodes.iter() {
        for (i, v) in node.vec.iter().enumerate() {
            means[i] += *v;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }
    let mut stds = vec![0.0; dim];
    for node in nodes.iter() {
        for (i, v) in node.vec.iter().enumerate() {
            let d = *v - means[i];
            stds[i] += d * d;
        }
    }
    for std in &mut stds {
        *std = (*std / n).sqrt().max(1e-5);
    }
    for node in nodes.iter_mut() {
        for i in 0..dim {
            node.vec[i] = (node.vec[i] - means[i]) / stds[i];
        }
        let norm = node.vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in &mut node.vec {
            *v /= norm;
        }
    }
}

fn build_anchor_positions(nodes: &[SampleNode]) -> Vec<Vec2> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let dim = nodes[0].vec.len();
    let mut a0 = vec![0.0; dim];
    let mut a1 = vec![0.0; dim];
    for i in 0..dim {
        a0[i] = ((i as f32 * 1.731).sin() + 0.1 * (i as f32 * 0.733).cos()) * (1.0 + i as f32 * 0.01);
        a1[i] = ((i as f32 * 2.173).cos() - 0.15 * (i as f32 * 0.417).sin()) * (1.0 + i as f32 * 0.01);
    }
    normalize_vec(&mut a0);
    orthogonalize_and_normalize(&mut a1, &a0);

    let mut raw: Vec<Vec2> = nodes
        .iter()
        .map(|n| {
            let x = dot(&n.vec, &a0);
            let y = dot(&n.vec, &a1);
            Vec2::new(x, y)
        })
        .collect();

    let min_x = raw.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    let max_x = raw.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let min_y = raw.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    let max_y = raw.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    let scale_x = (max_x - min_x).max(1e-5);
    let scale_y = (max_y - min_y).max(1e-5);

    for p in &mut raw {
        p.x = 0.15 + 0.70 * ((p.x - min_x) / scale_x);
        p.y = 0.15 + 0.70 * ((p.y - min_y) / scale_y);
    }
    raw
}

fn normalize_vec(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v {
        *x /= norm;
    }
}

fn orthogonalize_and_normalize(v: &mut [f32], basis: &[f32]) {
    let proj = dot(v, basis);
    for (x, b) in v.iter_mut().zip(basis.iter()) {
        *x -= proj * *b;
    }
    normalize_vec(v);
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn step_sim(nodes: &mut [SampleNode], neighbors: &[Vec<usize>], force: f32, damping: f32, dt: f32) {
    let n = nodes.len();
    if n == 0 {
        return;
    }

    let anchors = build_anchor_positions(nodes);
    let mut accels = vec![Vec2::ZERO; n];
    let repulsion_radius = (0.07 + 0.015 * force).clamp(0.06, 0.10);
    let center = Vec2::new(0.5, 0.5);

    for i in 0..n {
        let mut acc = Vec2::ZERO;

        for j in 0..n {
            if i == j {
                continue;
            }
            let delta = nodes[i].pos - nodes[j].pos;
            let dist = delta.length().max(1e-4);
            if dist < repulsion_radius {
                let strength = ((repulsion_radius - dist) / repulsion_radius).powi(2) * 0.020;
                acc += delta / dist * strength;
            }
        }

        for &j in neighbors.get(i).into_iter().flatten() {
            let sim = cosine(&nodes[i].vec, &nodes[j].vec).clamp(0.0, 1.0);
            if sim < 0.08 {
                continue;
            }
            let delta = nodes[j].pos - nodes[i].pos;
            let dist = delta.length().max(1e-4);
            let target = (0.11 - 0.065 * sim * force).clamp(0.03, 0.12);
            let spring = (dist - target) * (0.030 + 0.055 * sim * force);
            acc += delta / dist * spring;
        }

        acc += (anchors[i] - nodes[i].pos) * (0.020 + 0.020 * force);
        acc += (center - nodes[i].pos) * 0.0035;

        let max_acc = 0.03;
        let len = acc.length();
        if len > max_acc {
            acc = acc / len * max_acc;
        }
        accels[i] = acc;
    }

    for i in 0..n {
        nodes[i].vel = nodes[i].vel * damping + accels[i] * dt * 60.0;
        let max_vel = 0.015;
        let len = nodes[i].vel.length();
        if len > max_vel {
            nodes[i].vel = nodes[i].vel / len * max_vel;
        }
        nodes[i].pos += nodes[i].vel;
        nodes[i].pos.x = nodes[i].pos.x.clamp(0.03, 0.97);
        nodes[i].pos.y = nodes[i].pos.y.clamp(0.03, 0.97);
    }
}

fn build_neighbors(nodes: &[SampleNode], k: usize) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let mut out = vec![Vec::new(); n];
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&nodes[i].vec, &nodes[j].vec)))
            .filter(|(_, sim)| *sim > 0.05)
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (j, _) in sims.into_iter().take(k) {
            out[i].push(j);
        }
    }
    out
}

fn max_speed(nodes: &[SampleNode]) -> f32 {
    nodes.iter().map(|n| n.vel.length()).fold(0.0, |a, b| a.max(b))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = (a.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
    let nb = (b.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
    dot / (na * nb)
}

fn load_audio_mono(path: &Path) -> Result<Vec<f32>> {
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
        src.convert_samples::<f32>().take(44_100 * 12).collect::<Vec<f32>>()
    };

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
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                reader
                    .samples::<i16>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect()
            } else {
                reader
                    .samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i32::MAX as f32)
                    .collect()
            }
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    let channels = spec.channels.max(1) as usize;
    let mut mono = Vec::with_capacity(samples.len() / channels.max(1));
    for chunk in samples.chunks(channels) {
        let avg = chunk.iter().copied().sum::<f32>() / channels as f32;
        mono.push(avg);
    }
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
