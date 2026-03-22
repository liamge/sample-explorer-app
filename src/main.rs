use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2, ViewportBuilder};
use eframe::{App, Frame};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::prelude::*;
use rodio::{OutputStream, Sink, Source};
use rustfft::{num_complex::Complex, FftPlanner};

static AUDIO_EXT: Lazy<Vec<&str>> =
    Lazy::new(|| vec!["wav", "aiff", "aif", "flac", "ogg", "mp3"]);
static AUDIO_CACHE: Lazy<RwLock<HashMap<PathBuf, Vec<f32>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

const FIXED_DAMPING: f32 = 0.78;
const MIN_ZOOM: f32 = 24.0;
const MAX_ZOOM: f32 = 2200.0;
const MAX_ANALYSIS_SECONDS: f32 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimbreClass {
    Vocal,
    Drum,
    Tonal,
    Texture,
}

#[derive(Clone)]
struct SampleNode {
    path: PathBuf,
    directory_group: String,
    duration_s: f32,
    rms: f32,
    centroid: f32,
    zcr: f32,
    flatness: f32,
    attack: f32,
    vec: Vec<f32>,
    pos: Vec2,
    vel: Vec2,
    anchor: Vec2,
    color: Color32,
    timbre: TimbreClass,
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
    status: String,
    last_layout_ms: u128,
    sim_active: bool,
    neighbor_lists: Vec<Vec<usize>>,
    settle_frames: usize,
    selected: Option<usize>,
    search: String,
    directory_legend: Vec<(String, Color32)>,
    camera_center: Vec2,
    zoom: f32,
}

impl ExplorerApp {
    fn new() -> Self {
        Self {
            folder: None,
            nodes: Vec::new(),
            player: AudioPlayer::new().ok(),
            cluster_force: 1.0,
            status: "Open a folder".into(),
            last_layout_ms: 0,
            sim_active: false,
            neighbor_lists: Vec::new(),
            settle_frames: 0,
            selected: None,
            search: String::new(),
            directory_legend: Vec::new(),
            camera_center: Vec2::ZERO,
            zoom: 360.0,
        }
    }

    fn apply_style(ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.override_text_color = Some(Color32::from_rgb(231, 236, 242));
        style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(20, 26, 34);
        style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(27, 35, 46);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(38, 49, 63);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(50, 71, 92);
        style.visuals.widgets.open.bg_fill = Color32::from_rgb(34, 45, 58);
        style.visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(220, 228, 237);
        style.visuals.selection.bg_fill = Color32::from_rgb(74, 128, 184);
        style.visuals.panel_fill = Color32::from_rgb(14, 18, 24);
        style.visuals.window_fill = Color32::from_rgb(18, 24, 31);
        style.visuals.extreme_bg_color = Color32::from_rgb(11, 15, 20);
        style.visuals.faint_bg_color = Color32::from_rgb(21, 28, 36);
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(10.0, 6.0);
        style.spacing.menu_margin = egui::Margin::same(10.0);
        style.visuals.window_rounding = egui::Rounding::same(10.0);
        style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.active.rounding = egui::Rounding::same(8.0);
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(24.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
        );
        ctx.set_style(style);
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
        standardize_feature_vectors(&mut nodes);
        self.neighbor_lists = build_neighbors(&nodes, 10);
        apply_embedding_and_labels(folder, &self.neighbor_lists, &mut nodes);
        self.last_layout_ms = start.elapsed().as_millis();
        self.status = format!(
            "Loaded {} samples ({} ms, damping {:.2})",
            nodes.len(),
            self.last_layout_ms,
            FIXED_DAMPING
        );
        self.nodes = nodes;
        self.directory_legend = collect_directory_legend(&self.nodes);
        self.selected = None;
        self.restart_sim();
        self.fit_camera_to_nodes();
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
        for n in &mut self.nodes {
            n.pos = n.anchor + Vec2::new((rand_unit() - 0.5) * 0.12, (rand_unit() - 0.5) * 0.12);
            n.vel = Vec2::ZERO;
        }
        self.sim_active = !self.nodes.is_empty();
        self.settle_frames = 0;
    }

    fn fit_camera_to_nodes(&mut self) {
        if self.nodes.is_empty() {
            self.camera_center = Vec2::ZERO;
            self.zoom = 360.0;
            return;
        }

        let (min, max) = node_bounds(&self.nodes);
        self.camera_center = (min + max) * 0.5;
        let size = max - min;
        let extent = size.x.max(size.y).max(1.0);
        self.zoom = (650.0 / extent).clamp(MIN_ZOOM, MAX_ZOOM);
    }
}

impl App for ExplorerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        Self::apply_style(ctx);
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(6.0);
            egui::Frame::none()
                .fill(Color32::from_rgb(17, 23, 31))
                .rounding(egui::Rounding::same(10.0))
                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.vertical(|ui| {
                            ui.heading("Sample Explorer");
                            ui.label(egui::RichText::new("Similarity map for audio samples").color(Color32::from_gray(165)));
                        });
                        ui.add_space(16.0);
                        if ui.button("Open Folder").clicked() {
                            self.choose_folder();
                        }
                        if ui.button("Re-center").clicked() {
                            self.fit_camera_to_nodes();
                        }
                        ui.separator();
                        ui.vertical(|ui| {
                            ui.label("Cluster Strength");
                            if ui
                                .add(egui::Slider::new(&mut self.cluster_force, 0.2..=3.6).show_value(true))
                                .changed()
                            {
                                self.restart_sim();
                            }
                        });
                        ui.separator();
                        ui.vertical(|ui| {
                            ui.label("Layout");
                            ui.label(format!("{} ms", self.last_layout_ms));
                        });
                        ui.vertical(|ui| {
                            ui.label("Damping");
                            ui.label(format!("{:.2}", FIXED_DAMPING));
                        });
                        ui.separator();
                        ui.vertical(|ui| {
                            ui.label("Status");
                            ui.label(&self.status);
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new("Drag to pan, scroll to zoom").color(Color32::from_gray(150)));
                        });
                    });
                });
            ui.add_space(6.0);
        });

        egui::SidePanel::right("inspector")
            .resizable(false)
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(Color32::from_rgb(18, 24, 31))
                    .rounding(egui::Rounding::same(12.0))
                    .inner_margin(egui::Margin::same(14.0))
                    .show(ui, |ui| {
                        ui.heading("Inspector");
                        ui.label(egui::RichText::new("Search, inspect, and compare directory groups").color(Color32::from_gray(160)));
                        ui.add_space(8.0);
                        ui.label("Search");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search)
                                .hint_text("Filter by filename")
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(10.0);
                        ui.label("Directories");
                        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                            for (group, color) in &self.directory_legend {
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(Vec2::new(12.0, 12.0), Sense::hover());
                                    ui.painter().rect_filled(rect, 4.0, *color);
                                    ui.label(group);
                                });
                            }
                        });
                        ui.separator();
                        if let Some(idx) = self.selected {
                            if let Some(node) = self.nodes.get(idx) {
                                ui.heading(
                                    node.path
                                        .file_name()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("Unknown"),
                                );
                                ui.label(&node.directory_group);
                                ui.add_space(6.0);
                                ui.label(format!("Length {:.2}s", node.duration_s));
                                ui.label(format!("RMS {:.3}", node.rms));
                                ui.label(format!("Centroid {:.0} Hz", node.centroid));
                                ui.label(format!("ZCR {:.3}", node.zcr));
                                ui.label(format!("Flatness {:.3}", node.flatness));
                                ui.label(format!("Attack {:.3}", node.attack));
                                if ui.button("Play Selected").clicked() {
                                    let path = node.path.clone();
                                    self.click_play(&path);
                                }
                            }
                        } else {
                            ui.label("Select a node to inspect it.");
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (response, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
            let rect = response.rect;
            draw_background(&painter, rect, self.camera_center, self.zoom);
            draw_graph(
                &self.nodes,
                &self.neighbor_lists,
                &painter,
                rect,
                &self.search,
                self.selected,
                self.camera_center,
                self.zoom,
            );

            if response.dragged() {
                let delta = ctx.input(|i| i.pointer.delta());
                self.camera_center -= delta / self.zoom;
                ctx.request_repaint();
            }

            if response.hovered() {
                let scroll = ctx.input(|i| i.raw_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let zoom_factor = (1.0 + scroll * 0.0015).clamp(0.8, 1.25);
                    if let Some(pointer) = response.hover_pos() {
                        let before = screen_to_world(rect, pointer, self.camera_center, self.zoom);
                        self.zoom = (self.zoom * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);
                        let after = screen_to_world(rect, pointer, self.camera_center, self.zoom);
                        self.camera_center += before - after;
                    } else {
                        self.zoom = (self.zoom * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);
                    }
                    ctx.request_repaint();
                }
            }

            if response.clicked() || response.double_clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    if let Some(idx) = pick_node(
                        &self.nodes,
                        pos,
                        rect,
                        &self.search,
                        self.camera_center,
                        self.zoom,
                    ) {
                        self.selected = Some(idx);
                        let path = self.nodes[idx].path.clone();
                        self.click_play(&path);
                    }
                }
            }
        });

        if self.sim_active {
            step_sim(
                &mut self.nodes,
                &self.neighbor_lists,
                self.cluster_force,
                FIXED_DAMPING,
            );
            let speed = max_speed(&self.nodes);
            if speed < 0.00045 {
                self.settle_frames += 1;
            } else {
                self.settle_frames = 0;
            }
            if self.settle_frames >= 45 {
                self.sim_active = false;
                self.status = format!("Settled · damping {:.2}", FIXED_DAMPING);
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            }
        }
    }
}

fn draw_background(painter: &egui::Painter, rect: Rect, center: Vec2, zoom: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(10, 14, 19));
    painter.rect_filled(
        rect.shrink(8.0),
        14.0,
        Color32::from_rgba_premultiplied(16, 22, 29, 235),
    );
    let grid = Color32::from_gray(36);
    let step_world = choose_grid_step(zoom);
    let top_left = screen_to_world(rect, rect.left_top(), center, zoom);
    let bottom_right = screen_to_world(rect, rect.right_bottom(), center, zoom);
    let mut x = (top_left.x / step_world).floor() * step_world;
    while x <= bottom_right.x {
        let screen_x = world_to_screen(rect, Vec2::new(x, center.y), center, zoom).x;
        painter.line_segment(
            [Pos2::new(screen_x, rect.top()), Pos2::new(screen_x, rect.bottom())],
            Stroke::new(1.0, grid),
        );
        x += step_world;
    }
    let mut y = (top_left.y / step_world).floor() * step_world;
    while y <= bottom_right.y {
        let screen_y = world_to_screen(rect, Vec2::new(center.x, y), center, zoom).y;
        painter.line_segment(
            [Pos2::new(rect.left(), screen_y), Pos2::new(rect.right(), screen_y)],
            Stroke::new(1.0, grid),
        );
        y += step_world;
    }

    let axis = Color32::from_gray(74);
    if top_left.x <= 0.0 && bottom_right.x >= 0.0 {
        let screen_x = world_to_screen(rect, Vec2::new(0.0, center.y), center, zoom).x;
        painter.line_segment(
            [Pos2::new(screen_x, rect.top()), Pos2::new(screen_x, rect.bottom())],
            Stroke::new(1.5, axis),
        );
    }
    if top_left.y <= 0.0 && bottom_right.y >= 0.0 {
        let screen_y = world_to_screen(rect, Vec2::new(center.x, 0.0), center, zoom).y;
        painter.line_segment(
            [Pos2::new(rect.left(), screen_y), Pos2::new(rect.right(), screen_y)],
            Stroke::new(1.5, axis),
        );
    }
}

fn draw_graph(
    nodes: &[SampleNode],
    neighbors: &[Vec<usize>],
    painter: &egui::Painter,
    rect: Rect,
    search: &str,
    selected: Option<usize>,
    center: Vec2,
    zoom: f32,
) {
    let needle = search.trim().to_lowercase();
    for (i, node) in nodes.iter().enumerate() {
        if !matches_filter(node, &needle) {
            continue;
        }
        for &j in neighbors.get(i).into_iter().flatten() {
            if j <= i {
                continue;
            }
            if let Some(other) = nodes.get(j) {
                if !matches_filter(other, &needle) {
                    continue;
                }
                let sim = cosine(&node.vec, &other.vec);
                if sim < 0.58 {
                    continue;
                }
                let pa = world_to_screen(rect, node.pos, center, zoom);
                let pb = world_to_screen(rect, other.pos, center, zoom);
                let alpha = ((sim - 0.58) / 0.42).clamp(0.0, 1.0);
                let c = Color32::from_white_alpha((25.0 + 70.0 * alpha) as u8);
                painter.line_segment([pa, pb], Stroke::new(1.0, c));
            }
        }
    }

    for (i, node) in nodes.iter().enumerate() {
        if !matches_filter(node, &needle) {
            continue;
        }
        let p = world_to_screen(rect, node.pos, center, zoom);
        if !rect.expand(24.0).contains(p) {
            continue;
        }
        let radius = if Some(i) == selected { 9.5_f32 } else { 6.5_f32 };
        if Some(i) == selected {
            painter.circle_filled(p, radius + 3.0, Color32::from_white_alpha(26));
        }
        painter.circle_filled(p, radius, node.color);
        painter.circle_stroke(p, radius, Stroke::new(1.0, Color32::from_white_alpha(35)));
    }
}

fn matches_filter(node: &SampleNode, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    node.path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase().contains(needle))
        .unwrap_or(false)
}

fn pick_node(
    nodes: &[SampleNode],
    mouse: Pos2,
    rect: Rect,
    search: &str,
    center: Vec2,
    zoom: f32,
) -> Option<usize> {
    let needle = search.trim().to_lowercase();
    let mut best = None;
    let mut best_d = 18.0f32 * 18.0;
    for (i, n) in nodes.iter().enumerate() {
        if !matches_filter(n, &needle) {
            continue;
        }
        let p = world_to_screen(rect, n.pos, center, zoom);
        let d = p.distance_sq(mouse);
        if d < best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

fn world_to_screen(rect: Rect, world: Vec2, center: Vec2, zoom: f32) -> Pos2 {
    let delta = (world - center) * zoom;
    Pos2::new(rect.center().x + delta.x, rect.center().y + delta.y)
}

fn screen_to_world(rect: Rect, screen: Pos2, center: Vec2, zoom: f32) -> Vec2 {
    center + (screen - rect.center()) / zoom
}

fn choose_grid_step(zoom: f32) -> f32 {
    let pixels_per_step = 96.0;
    let target = (pixels_per_step / zoom).max(1e-4);
    let pow10 = 10.0f32.powf(target.log10().floor());
    for factor in [1.0, 2.0, 5.0, 10.0] {
        let step = pow10 * factor;
        if step >= target {
            return step;
        }
    }
    pow10 * 10.0
}

fn feature_node(path: &Path) -> Result<SampleNode> {
    let samples = load_audio_mono(path)?;
    if samples.is_empty() {
        return Err(anyhow!("empty"));
    }
    let duration_s = (samples.len() as f32 / 44_100.0).clamp(0.0, MAX_ANALYSIS_SECONDS);

    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
    let zcr = samples
        .windows(2)
        .filter(|w| (w[0].is_sign_positive()) != (w[1].is_sign_positive()))
        .count() as f32
        / samples.len().max(1) as f32;

    let (centroid, rolloff, flatness, attack, flux, crest, band_stats) = timbre_features(&samples, 44_100);
    let mut vec = vec![
        duration_s / MAX_ANALYSIS_SECONDS,
        rms,
        zcr,
        centroid / 12_000.0,
        rolloff / 18_000.0,
        flatness,
        attack,
        flux,
        crest / 20.0,
    ];
    vec.extend(band_stats);

    Ok(SampleNode {
        path: path.to_path_buf(),
        directory_group: String::new(),
        duration_s,
        rms,
        centroid,
        zcr,
        flatness,
        attack,
        vec,
        pos: Vec2::ZERO,
        vel: Vec2::ZERO,
        anchor: Vec2::new(0.5, 0.5),
        color: Color32::from_rgb(110, 227, 255),
        timbre: TimbreClass::Texture,
    })
}

fn timbre_features(samples: &[f32], sr: u32) -> (f32, f32, f32, f32, f32, f32, Vec<f32>) {
    let frame = 1024usize;
    let hop = 512usize;
    let nfft = 1024usize;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let window: Vec<f32> = (0..frame)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / frame as f32).cos())
        .collect();
    let edges = [0.0, 120.0, 300.0, 800.0, 2000.0, 5000.0, 12000.0, 22050.0];

    let mut centroids = Vec::new();
    let mut rolloffs = Vec::new();
    let mut flatnesses = Vec::new();
    let mut fluxes = Vec::new();
    let mut crests = Vec::new();
    let mut attacks = Vec::new();
    let mut band_accum = vec![0.0f32; edges.len() - 1];
    let mut prev_mag = vec![0.0f32; nfft / 2];
    let mut frame_count = 0.0f32;

    let mut start = 0usize;
    while start + frame <= samples.len().min(44_100 * 12) {
        let mut buf = vec![Complex::new(0.0f32, 0.0f32); nfft];
        let frame_slice = &samples[start..start + frame];
        let mut frame_rms = 0.0;
        let mut max_abs: f32 = 0.0;
        for i in 0..frame {
            let s = frame_slice[i] * window[i];
            frame_rms += s * s;
            max_abs = max_abs.max(s.abs());
            buf[i] = Complex::new(s, 0.0);
        }
        frame_rms = (frame_rms / frame as f32).sqrt();
        fft.process(&mut buf);

        let mags: Vec<f32> = buf[..nfft / 2].iter().map(|c| c.norm()).collect();
        let freqs: Vec<f32> = (0..(nfft / 2)).map(|i| i as f32 * sr as f32 / nfft as f32).collect();
        let sum_mag = mags.iter().sum::<f32>().max(1e-6);
        let centroid = freqs
            .iter()
            .zip(mags.iter())
            .map(|(f, m)| f * m)
            .sum::<f32>()
            / sum_mag;
        let mut cumulative = 0.0;
        let mut rolloff = sr as f32 / 2.0;
        for (f, m) in freqs.iter().zip(mags.iter()) {
            cumulative += *m;
            if cumulative >= sum_mag * 0.85 {
                rolloff = *f;
                break;
            }
        }
        let mean_mag = sum_mag / mags.len().max(1) as f32;
        let geo_mag = (mags.iter().map(|m| m.max(1e-12).ln()).sum::<f32>() / mags.len().max(1) as f32).exp();
        let flatness = (geo_mag / mean_mag.max(1e-6)).clamp(0.0, 1.0);
        let flux = mags
            .iter()
            .zip(prev_mag.iter())
            .map(|(a, b)| (a - b).max(0.0))
            .sum::<f32>()
            / mags.len().max(1) as f32;
        let crest = max_abs / frame_rms.max(1e-4);
        let attack = frame_slice
            .windows(2)
            .map(|w| (w[1].abs() - w[0].abs()).max(0.0))
            .sum::<f32>()
            / frame as f32;

        for band in 0..(edges.len() - 1) {
            let lo = edges[band];
            let hi = edges[band + 1];
            let mut sum = 0.0;
            let mut count = 0.0;
            for (f, m) in freqs.iter().zip(mags.iter()) {
                if *f >= lo && *f < hi {
                    sum += *m;
                    count += 1.0;
                }
            }
            band_accum[band] += if count > 0.0 { (sum / count).ln_1p() } else { 0.0 };
        }

        centroids.push(centroid);
        rolloffs.push(rolloff);
        flatnesses.push(flatness);
        fluxes.push(flux);
        crests.push(crest);
        attacks.push(attack);
        prev_mag.copy_from_slice(&mags);
        frame_count += 1.0;
        start += hop;
    }

    if frame_count <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec![0.0; edges.len() - 1]);
    }

    let band_stats = band_accum.into_iter().map(|v| v / frame_count).collect::<Vec<_>>();
    (
        mean(&centroids),
        mean(&rolloffs),
        mean(&flatnesses),
        mean(&attacks),
        mean(&fluxes),
        mean(&crests),
        band_stats,
    )
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

fn standardize_feature_vectors(nodes: &mut [SampleNode]) {
    if nodes.is_empty() {
        return;
    }
    let dim = nodes[0].vec.len();
    let mut means = vec![0.0f32; dim];
    for node in nodes.iter() {
        for (i, v) in node.vec.iter().enumerate() {
            means[i] += *v;
        }
    }
    for m in &mut means {
        *m /= nodes.len() as f32;
    }

    let mut stds = vec![0.0f32; dim];
    for node in nodes.iter() {
        for (i, v) in node.vec.iter().enumerate() {
            let d = *v - means[i];
            stds[i] += d * d;
        }
    }
    for s in &mut stds {
        *s = (*s / nodes.len() as f32).sqrt().max(1e-5);
    }

    for node in nodes.iter_mut() {
        for i in 0..dim {
            node.vec[i] = (node.vec[i] - means[i]) / stds[i];
        }
    }
}

fn apply_embedding_and_labels(folder: &Path, neighbors: &[Vec<usize>], nodes: &mut [SampleNode]) {
    if nodes.is_empty() {
        return;
    }
    let embedding = similarity_embedding_2d(nodes, neighbors);
    let colors = assign_colors(folder, nodes);
    for (idx, node) in nodes.iter_mut().enumerate() {
        let base = embedding.get(idx).copied().unwrap_or([0.0, 0.0]);
        let timbre = classify_timbre(node);
        let region_offset = match timbre {
            TimbreClass::Vocal => Vec2::new(0.28, -0.22),
            TimbreClass::Drum => Vec2::new(0.32, 0.26),
            TimbreClass::Tonal => Vec2::new(-0.24, -0.24),
            TimbreClass::Texture => Vec2::new(-0.30, 0.28),
        };
        let anchor = Vec2::new(base[0], base[1]) * 3.2 + region_offset * 0.5;
        node.anchor = anchor;
        node.pos = node.anchor;
        node.vel = Vec2::ZERO;
        node.directory_group = directory_group_name(folder, &node.path);
        node.timbre = timbre;
        node.color = colors
            .get(&node.directory_group)
            .copied()
            .unwrap_or(Color32::from_rgb(110, 227, 255));
    }
}

fn classify_timbre(node: &SampleNode) -> TimbreClass {
    let centroid_n = node.centroid / 6000.0;
    let vocal_score = (1.2 - (centroid_n - 0.42).abs() * 2.2) + (0.30 - node.zcr).max(0.0) + (0.45 - node.flatness).max(0.0);
    let drum_score = node.attack * 12.0 + node.zcr * 4.0 + node.flatness * 2.5 + (node.centroid / 7000.0);
    let tonal_score = (0.55 - node.flatness).max(0.0) * 3.5 + (0.25 - node.zcr).max(0.0) * 4.0 + (0.9 - node.attack * 5.0).max(0.0);
    let texture_score = node.flatness * 3.0 + (node.centroid / 8000.0) + (node.attack * 2.0).min(1.5);

    let mut best = (TimbreClass::Vocal, vocal_score);
    for candidate in [
        (TimbreClass::Drum, drum_score),
        (TimbreClass::Tonal, tonal_score),
        (TimbreClass::Texture, texture_score),
    ] {
        if candidate.1 > best.1 {
            best = candidate;
        }
    }
    best.0
}

fn pca_embedding_2d(nodes: &[SampleNode]) -> Vec<[f32; 2]> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let n = nodes.len();
    let d = nodes[0].vec.len();
    if d == 0 {
        return vec![[0.5, 0.5]; n];
    }

    let mut cov = vec![vec![0.0f32; d]; d];
    for node in nodes {
        for i in 0..d {
            for j in i..d {
                cov[i][j] += node.vec[i] * node.vec[j];
            }
        }
    }
    let denom = (n as f32 - 1.0).max(1.0);
    for i in 0..d {
        for j in i..d {
            cov[i][j] /= denom;
            cov[j][i] = cov[i][j];
        }
    }

    let v1 = power_iteration(&cov, 48);
    let lambda1 = quad_form(&cov, &v1).max(1e-6);
    let mut cov2 = cov.clone();
    for i in 0..d {
        for j in 0..d {
            cov2[i][j] -= lambda1 * v1[i] * v1[j];
        }
    }
    let v2 = power_iteration(&cov2, 48);

    let mut pts = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for node in nodes {
        let x = dot(&node.vec, &v1);
        let y = dot(&node.vec, &v2);
        xs.push(x);
        ys.push(y);
        pts.push([x, y]);
    }
    let (min_x, max_x) = min_max(&xs);
    let (min_y, max_y) = min_max(&ys);
    for p in &mut pts {
        p[0] = remap(p[0], min_x, max_x);
        p[1] = remap(p[1], min_y, max_y);
    }
    pts
}

fn similarity_embedding_2d(nodes: &[SampleNode], neighbors: &[Vec<usize>]) -> Vec<[f32; 2]> {
    if nodes.is_empty() {
        return Vec::new();
    }
    if nodes.len() == 1 {
        return vec![[0.0, 0.0]];
    }

    let mut pts = pca_embedding_2d(nodes)
        .into_iter()
        .map(|p| Vec2::new((p[0] - 0.5) * 2.0, (p[1] - 0.5) * 2.0))
        .collect::<Vec<_>>();

    let n = nodes.len();
    let pair_samples = n.min(80);
    for _ in 0..260 {
        let mut acc = vec![Vec2::ZERO; n];

        for i in 0..n {
            for &j in neighbors.get(i).into_iter().flatten() {
                if j == i {
                    continue;
                }
                let delta = pts[j] - pts[i];
                let dist = delta.length().max(1e-4);
                let dir = delta / dist;
                let sim = cosine(&nodes[i].vec, &nodes[j].vec).clamp(-1.0, 1.0);
                let sim01 = (sim + 1.0) * 0.5;
                let target = 0.14 + (1.0 - sim01).powf(1.8) * 1.85;
                let spring = (dist - target) * (0.040 + 0.085 * sim01);
                acc[i] += dir * spring;
            }
        }

        for i in 0..n {
            for offset in 1..=pair_samples {
                let j = (i + offset * 17) % n;
                if j == i {
                    continue;
                }
                let delta = pts[j] - pts[i];
                let dist_sq = delta.length_sq().max(0.02);
                let dist = dist_sq.sqrt();
                let dir = delta / dist;
                let sim = cosine(&nodes[i].vec, &nodes[j].vec).clamp(-1.0, 1.0);
                let sim01 = (sim + 1.0) * 0.5;
                let repulsion = (1.0 - sim01).powf(1.3) * 0.008 / dist_sq;
                acc[i] -= dir * repulsion;
            }
        }

        let centroid = pts.iter().copied().fold(Vec2::ZERO, |sum, p| sum + p) / n as f32;
        for i in 0..n {
            acc[i] += (Vec2::ZERO - pts[i]) * 0.0025;
            pts[i] += clamp_vec(acc[i], 0.08);
            pts[i] -= centroid * 0.003;
        }
    }

    pts.into_iter().map(|p| [p.x, p.y]).collect()
}

fn power_iteration(cov: &[Vec<f32>], iters: usize) -> Vec<f32> {
    let d = cov.len();
    let mut v = (0..d)
        .map(|i| ((i as f32 * 1.37).sin() + 1.0).max(0.01))
        .collect::<Vec<_>>();
    normalize_in_place(&mut v);
    for _ in 0..iters {
        let mut next = vec![0.0f32; d];
        for i in 0..d {
            next[i] = dot(&cov[i], &v);
        }
        normalize_in_place(&mut next);
        v = next;
    }
    v
}

fn quad_form(m: &[Vec<f32>], v: &[f32]) -> f32 {
    let mut tmp = vec![0.0f32; v.len()];
    for i in 0..v.len() {
        tmp[i] = dot(&m[i], v);
    }
    dot(&tmp, v)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn min_max(xs: &[f32]) -> (f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &x in xs {
        min_v = min_v.min(x);
        max_v = max_v.max(x);
    }
    if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < 1e-6 {
        (0.0, 1.0)
    } else {
        (min_v, max_v)
    }
}

fn remap(v: f32, lo: f32, hi: f32) -> f32 {
    if (hi - lo).abs() < 1e-6 {
        0.5
    } else {
        ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
    }
}

fn assign_colors(folder: &Path, nodes: &[SampleNode]) -> HashMap<String, Color32> {
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
    for n in nodes {
        let group = directory_group_name(folder, &n.path);
        let color = *group_color.entry(group.clone()).or_insert_with(|| {
            let mut hasher = DefaultHasher::new();
            group.hash(&mut hasher);
            let idx = (hasher.finish() % palette.len() as u64) as usize;
            palette[idx]
        });
        group_color.insert(group, color);
    }
    group_color
}

fn directory_group_name(folder: &Path, path: &Path) -> String {
    path.strip_prefix(folder)
        .ok()
        .and_then(|p| p.parent())
        .and_then(|p| p.components().next())
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "root".into())
}

fn collect_directory_legend(nodes: &[SampleNode]) -> Vec<(String, Color32)> {
    let mut entries = HashMap::<String, Color32>::new();
    for node in nodes {
        entries
            .entry(node.directory_group.clone())
            .or_insert(node.color);
    }
    let mut entries = entries.into_iter().collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn rand_unit() -> f32 {
    use rand::Rng;
    static RNG: Lazy<parking_lot::Mutex<rand::rngs::StdRng>> =
        Lazy::new(|| parking_lot::Mutex::new(rand::SeedableRng::seed_from_u64(42)));
    let mut rng = RNG.lock();
    rng.gen::<f32>()
}

fn step_sim(nodes: &mut [SampleNode], neighbors: &[Vec<usize>], force: f32, damping: f32) {
    let n = nodes.len();
    if n == 0 {
        return;
    }

    let mut accs = vec![Vec2::ZERO; n];
    let force_scale = force.clamp(0.2, 3.6);
    let separation_exp = ((force_scale - 1.0) * 0.72).exp();
    let repulsion_radius = 0.34f32 + 0.12 * separation_exp.min(5.0);

    for i in 0..n {
        let anchor_pull = 0.010 / separation_exp.powf(0.32);
        accs[i] += (nodes[i].anchor - nodes[i].pos) * anchor_pull;

        for &j in neighbors.get(i).into_iter().flatten() {
            let delta = nodes[j].pos - nodes[i].pos;
            let dist = delta.length().max(1e-4);
            let dir = delta / dist;
            let sim = cosine(&nodes[i].vec, &nodes[j].vec).clamp(-1.0, 1.0);
            let sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0);
            let base_target = 0.10 + 0.92 * (1.0 - sim01).powf(1.7);
            let separation_bias = (1.0 - sim01).powf(1.35);
            let target = base_target * separation_exp.powf(0.35 + separation_bias * 1.55);
            let spring = (dist - target) * (0.010 + 0.030 * sim01) / separation_exp.powf(0.15);
            accs[i] += dir * spring;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let delta = nodes[j].pos - nodes[i].pos;
            let dist = delta.length().max(1e-4);
            if dist < repulsion_radius {
                let dir = delta / dist;
                let strength = (repulsion_radius - dist) / repulsion_radius
                    * (0.028 + 0.020 * separation_exp.min(4.5));
                accs[i] -= dir * strength;
                accs[j] += dir * strength;
            }
        }
    }

    for i in 0..n {
        let acc = clamp_vec(accs[i], 0.028);
        nodes[i].vel = (nodes[i].vel + acc) * damping;
        nodes[i].vel = clamp_vec(nodes[i].vel, 0.042);
        nodes[i].pos += nodes[i].vel;
    }
}

fn node_bounds(nodes: &[SampleNode]) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for node in nodes {
        min.x = min.x.min(node.pos.x);
        min.y = min.y.min(node.pos.y);
        max.x = max.x.max(node.pos.x);
        max.y = max.y.max(node.pos.y);
    }
    if !min.x.is_finite() || !min.y.is_finite() || !max.x.is_finite() || !max.y.is_finite() {
        (Vec2::ZERO, Vec2::ZERO)
    } else {
        (min, max)
    }
}

fn clamp_vec(v: Vec2, max_len: f32) -> Vec2 {
    let len = v.length();
    if len > max_len && len > 0.0 {
        v / len * max_len
    } else {
        v
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
        src.convert_samples::<f32>()
            .take(44_100 * 12)
            .collect::<Vec<f32>>()
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

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        (_, 8) => reader
            .samples::<i8>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i8::MAX as f32)
            .collect(),
        (_, 16) => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect(),
        (_, 24 | 32) => reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i32::MAX as f32)
            .collect(),
        _ => reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i32::MAX as f32)
            .collect(),
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
    let viewport = ViewportBuilder::default().with_inner_size([1280.0, 820.0]);
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
