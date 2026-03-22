use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
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
use serde::{Deserialize, Serialize};

static AUDIO_EXT: Lazy<Vec<&str>> = Lazy::new(|| vec!["wav", "aiff", "aif", "flac", "ogg", "mp3"]);
static AUDIO_CACHE: Lazy<RwLock<HashMap<PathBuf, AudioClip>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

const FIXED_DAMPING: f32 = 0.78;
const MIN_ZOOM: f32 = 24.0;
const MAX_ZOOM: f32 = 2200.0;
const MAX_ANALYSIS_SECONDS: f32 = 12.0;
const WAVEFORM_BINS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimbreClass {
    Vocal,
    Drum,
    Tonal,
    Texture,
}

#[derive(Clone)]
struct AudioClip {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    duration_s: f32,
    peak: f32,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct PersistedItemState {
    favorite: bool,
    tags: Vec<String>,
    collections: Vec<String>,
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedLibraryState {
    items: HashMap<String, PersistedItemState>,
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedAppState {
    default_folder: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportPreset {
    SourceMono16,
    Dawk48Mono24,
    DigitaktMono16,
    Sp404Mono16,
    MpcMono16,
}

impl ExportPreset {
    fn label(self) -> &'static str {
        match self {
            Self::SourceMono16 => "Source mono 16-bit",
            Self::Dawk48Mono24 => "DAW 48k 24-bit",
            Self::DigitaktMono16 => "Digitakt 48k 16-bit",
            Self::Sp404Mono16 => "SP-404 48k 16-bit",
            Self::MpcMono16 => "MPC 44.1k 16-bit",
        }
    }

    fn sample_rate(self, fallback: u32) -> u32 {
        match self {
            Self::SourceMono16 => fallback,
            Self::Dawk48Mono24 | Self::DigitaktMono16 | Self::Sp404Mono16 => 48_000,
            Self::MpcMono16 => 44_100,
        }
    }

    fn bits_per_sample(self) -> u16 {
        match self {
            Self::Dawk48Mono24 => 24,
            _ => 16,
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::SourceMono16 => "src16",
            Self::Dawk48Mono24 => "daw48k24",
            Self::DigitaktMono16 => "digitakt48k",
            Self::Sp404Mono16 => "sp40448k",
            Self::MpcMono16 => "mpc44k",
        }
    }
}

#[derive(Clone)]
struct SampleNode {
    path: PathBuf,
    directory_group: String,
    duration_s: f32,
    sample_rate: u32,
    channels: u16,
    rms: f32,
    centroid: f32,
    zcr: f32,
    flatness: f32,
    attack: f32,
    bpm: Option<f32>,
    transient_count: usize,
    loop_score: f32,
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
    library_state: PersistedLibraryState,
    app_state: PersistedAppState,
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
    selected_clip: Option<AudioClip>,
    waveform_cache: Vec<(f32, f32)>,
    transient_times: Vec<f32>,
    trim_start: f32,
    trim_end: f32,
    fade_in_ms: f32,
    fade_out_ms: f32,
    normalize_export: bool,
    export_preset: ExportPreset,
    tags_input: String,
    collections_input: String,
}

impl ExplorerApp {
    fn new() -> Self {
        let mut app = Self {
            folder: None,
            nodes: Vec::new(),
            player: AudioPlayer::new().ok(),
            library_state: PersistedLibraryState::default(),
            app_state: Self::load_app_state(),
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
            selected_clip: None,
            waveform_cache: Vec::new(),
            transient_times: Vec::new(),
            trim_start: 0.0,
            trim_end: 1.0,
            fade_in_ms: 0.0,
            fade_out_ms: 0.0,
            normalize_export: true,
            export_preset: ExportPreset::SourceMono16,
            tags_input: String::new(),
            collections_input: String::new(),
        };
        app.load_default_folder_on_startup();
        app
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
            self.load_library_state();
            self.status = "Scanning…".into();
            self.scan_folder();
        }
    }

    fn app_state_path() -> PathBuf {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".sample_explorer_app_state.json")
    }

    fn load_app_state() -> PersistedAppState {
        fs::read_to_string(Self::app_state_path())
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn save_app_state(&self) {
        if let Ok(text) = serde_json::to_string_pretty(&self.app_state) {
            let _ = fs::write(Self::app_state_path(), text);
        }
    }

    fn load_default_folder_on_startup(&mut self) {
        let Some(folder) = self.app_state.default_folder.clone() else {
            return;
        };
        if folder.is_dir() {
            self.folder = Some(folder);
            self.load_library_state();
            self.status = "Scanning default folder…".into();
            self.scan_folder();
        } else {
            self.status = format!(
                "Default folder not found: {}",
                folder.to_string_lossy()
            );
        }
    }

    fn save_current_folder_as_default(&mut self) {
        let Some(folder) = self.folder.clone() else {
            self.status = "Open a folder before setting a default".into();
            return;
        };
        self.app_state.default_folder = Some(folder.clone());
        self.save_app_state();
        self.status = format!("Default folder set to {}", folder.display());
    }

    fn clear_default_folder(&mut self) {
        self.app_state.default_folder = None;
        self.save_app_state();
        self.status = "Cleared default startup folder".into();
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
        let mut nodes: Vec<SampleNode> = files
            .par_iter()
            .filter_map(|p| feature_node(p).ok())
            .collect();
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
        self.selected_clip = None;
        self.waveform_cache.clear();
        self.transient_times.clear();
        self.restart_sim();
        self.fit_camera_to_nodes();
    }

    fn play_clip_region(&mut self, path: &Path, clip: &AudioClip, region_only: bool) {
        let processed = processed_audio_for_export(
            clip,
            self.trim_start,
            self.trim_end,
            self.fade_in_ms,
            self.fade_out_ms,
            self.normalize_export,
            self.export_preset.sample_rate(clip.sample_rate),
        );
        let playback = if region_only {
            processed
        } else {
            clip.samples.clone()
        };
        if playback.is_empty() {
            self.status = "Nothing to play in current region".into();
            return;
        }
        if let Some(player) = &self.player {
            let rate = if region_only {
                self.export_preset.sample_rate(clip.sample_rate)
            } else {
                clip.sample_rate
            };
            let _ = player.play(&playback, rate);
            self.status = format!(
                "Previewing {}",
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sample")
            );
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

    fn resume_sim(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        self.sim_active = true;
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

    fn state_path(&self) -> Option<PathBuf> {
        self.folder
            .as_ref()
            .map(|folder| folder.join(".sample_explorer_state.json"))
    }

    fn load_library_state(&mut self) {
        let Some(path) = self.state_path() else {
            return;
        };
        self.library_state = fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
    }

    fn save_library_state(&self) {
        let Some(path) = self.state_path() else {
            return;
        };
        if let Ok(text) = serde_json::to_string_pretty(&self.library_state) {
            let _ = fs::write(path, text);
        }
    }

    fn relative_key(&self, path: &Path) -> String {
        self.folder
            .as_ref()
            .and_then(|folder| path.strip_prefix(folder).ok())
            .unwrap_or(path)
            .to_string_lossy()
            .to_string()
    }

    fn selected_item_state(&self) -> PersistedItemState {
        let Some(idx) = self.selected else {
            return PersistedItemState::default();
        };
        let Some(node) = self.nodes.get(idx) else {
            return PersistedItemState::default();
        };
        self.library_state
            .items
            .get(&self.relative_key(&node.path))
            .cloned()
            .unwrap_or_default()
    }

    fn update_selected_item_state(&mut self, mutate: impl FnOnce(&mut PersistedItemState)) {
        let Some(idx) = self.selected else { return };
        let Some(node) = self.nodes.get(idx) else {
            return;
        };
        let key = self.relative_key(&node.path);
        let state = self.library_state.items.entry(key).or_default();
        mutate(state);
        self.save_library_state();
    }

    fn sync_selected_text_fields(&mut self) {
        let state = self.selected_item_state();
        self.tags_input = state.tags.join(", ");
        self.collections_input = state.collections.join(", ");
    }

    fn commit_selected_text_fields(&mut self) {
        let tags = parse_csv_field(&self.tags_input);
        let collections = parse_csv_field(&self.collections_input);
        self.update_selected_item_state(|state| {
            state.tags = tags;
            state.collections = collections;
        });
    }

    fn select_node(&mut self, idx: usize, autoplay: bool, region_only: bool) {
        self.selected = Some(idx);
        let Some(node) = self.nodes.get(idx).cloned() else {
            return;
        };
        let Ok(clip) = load_audio_clip(&node.path) else {
            self.status = format!(
                "Failed to load {:?}",
                node.path.file_name().unwrap_or_default()
            );
            return;
        };
        self.waveform_cache = build_waveform_overview(&clip.samples, WAVEFORM_BINS);
        let analysis = resample_linear(
            &clip.samples,
            clip.sample_rate,
            44_100,
            Some((44_100.0 * MAX_ANALYSIS_SECONDS) as usize),
        );
        let (transient_times, bpm, _) = detect_transients_and_bpm(&analysis, 44_100);
        self.selected_clip = Some(clip.clone());
        self.transient_times = transient_times;
        self.trim_start = 0.0;
        self.trim_end = 1.0;
        self.fade_in_ms = 0.0;
        self.fade_out_ms = 0.0;
        self.sync_selected_text_fields();
        if let Some(selected_node) = self.nodes.get_mut(idx) {
            selected_node.bpm = bpm.or(selected_node.bpm);
        }
        if autoplay {
            self.play_clip_region(&node.path, &clip, region_only);
        }
    }

    fn selected_clip_duration(&self) -> f32 {
        self.selected_clip
            .as_ref()
            .map(|clip| clip.duration_s)
            .unwrap_or(0.0)
    }

    fn export_selected(&mut self) {
        let Some(idx) = self.selected else { return };
        let Some(node) = self.nodes.get(idx) else {
            return;
        };
        let Some(clip) = self.selected_clip.as_ref() else {
            return;
        };
        let suggested = format!(
            "{}_{}.wav",
            node.path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("sample"),
            self.export_preset.suffix()
        );
        if let Some(path) = rfd::FileDialog::new().set_file_name(&suggested).save_file() {
            match export_clip_to_wav(
                clip,
                &path,
                self.trim_start,
                self.trim_end,
                self.fade_in_ms,
                self.fade_out_ms,
                self.normalize_export,
                self.export_preset,
            ) {
                Ok(()) => self.status = format!("Exported {}", path.display()),
                Err(err) => self.status = format!("Export failed: {err}"),
            }
        }
    }

    fn export_batch(&mut self, favorites_only: bool) {
        let Some(target_dir) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        let needle = self.search.trim().to_lowercase();
        let mut exported = 0usize;
        for node in &self.nodes {
            if !matches_filter(node, &needle) {
                continue;
            }
            let state = self
                .library_state
                .items
                .get(&self.relative_key(&node.path))
                .cloned()
                .unwrap_or_default();
            if favorites_only && !state.favorite {
                continue;
            }
            let Ok(clip) = load_audio_clip(&node.path) else {
                continue;
            };
            let name = format!(
                "{}_{}.wav",
                node.path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sample"),
                self.export_preset.suffix()
            );
            let path = target_dir.join(name);
            if export_clip_to_wav(&clip, &path, 0.0, 1.0, 0.0, 0.0, true, self.export_preset)
                .is_ok()
            {
                exported += 1;
            }
        }
        self.status = format!("Exported {exported} files to {}", target_dir.display());
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
                            ui.label(
                                egui::RichText::new("Similarity map for audio samples")
                                    .color(Color32::from_gray(165)),
                            );
                        });
                        ui.add_space(16.0);
                        if ui.button("Open Folder").clicked() {
                            self.choose_folder();
                        }
                        if ui.button("Set Default").clicked() {
                            self.save_current_folder_as_default();
                        }
                        if ui.button("Clear Default").clicked() {
                            self.clear_default_folder();
                        }
                        if ui.button("Re-center").clicked() {
                            self.fit_camera_to_nodes();
                        }
                        ui.separator();
                        ui.vertical(|ui| {
                            ui.label("Cluster Strength");
                            if ui
                                .add_sized(
                                    [220.0, 0.0],
                                    egui::Slider::new(&mut self.cluster_force, 0.1..=6.0)
                                        .show_value(true),
                                )
                                .changed()
                            {
                                self.resume_sim();
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
                            ui.label(
                                egui::RichText::new("Drag to pan, scroll to zoom")
                                    .color(Color32::from_gray(150)),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        });

        egui::SidePanel::right("inspector")
            .resizable(false)
            .default_width(340.0)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(Color32::from_rgb(18, 24, 31))
                    .rounding(egui::Rounding::same(12.0))
                    .inner_margin(egui::Margin::same(14.0))
                    .show(ui, |ui| {
                        ui.heading("Inspector");
                        ui.label(
                            egui::RichText::new("Search, inspect, and compare directory groups")
                                .color(Color32::from_gray(160)),
                        );
                        ui.add_space(8.0);
                        ui.label("Search");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search)
                                .hint_text("Filter by filename")
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(10.0);
                        ui.label("Directories");
                        egui::ScrollArea::vertical()
                            .max_height(180.0)
                            .show(ui, |ui| {
                                for (group, color) in &self.directory_legend {
                                    ui.horizontal(|ui| {
                                        let (rect, _) = ui.allocate_exact_size(
                                            Vec2::new(12.0, 12.0),
                                            Sense::hover(),
                                        );
                                        ui.painter().rect_filled(rect, 4.0, *color);
                                        ui.label(group);
                                    });
                                }
                            });
                        ui.separator();
                        if let Some(idx) = self.selected {
                            if let Some(node) = self.nodes.get(idx).cloned() {
                                let item_state = self.selected_item_state();
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    ui.heading(
                                        node.path
                                            .file_name()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("Unknown"),
                                    );
                                    ui.label(&node.directory_group);
                                    ui.add_space(6.0);
                                    ui.label(format!(
                                        "Length {:.2}s · {} Hz · {}ch",
                                        node.duration_s, node.sample_rate, node.channels
                                    ));
                                    if let Some(clip) = &self.selected_clip {
                                        ui.label(format!("Peak {:.3}", clip.peak));
                                    }
                                    ui.label(format!(
                                        "BPM {} · Transients {} · Loop {:.0}%",
                                        node.bpm
                                            .map(|b| format!("{b:.1}"))
                                            .unwrap_or_else(|| "n/a".into()),
                                        node.transient_count,
                                        node.loop_score * 100.0
                                    ));
                                    ui.label(format!("RMS {:.3}", node.rms));
                                    ui.label(format!("Centroid {:.0} Hz", node.centroid));
                                    ui.label(format!("ZCR {:.3}", node.zcr));
                                    ui.label(format!("Flatness {:.3}", node.flatness));
                                    ui.label(format!("Attack {:.3}", node.attack));
                                    ui.add_space(8.0);

                                    let mut favorite = item_state.favorite;
                                    if ui.checkbox(&mut favorite, "Favorite").changed() {
                                        self.update_selected_item_state(|state| {
                                            state.favorite = favorite
                                        });
                                    }
                                    ui.label("Tags");
                                    if ui
                                        .add(
                                            egui::TextEdit::singleline(&mut self.tags_input)
                                                .hint_text("kick, dusty, bright"),
                                        )
                                        .changed()
                                    {
                                        self.commit_selected_text_fields();
                                    }
                                    ui.label("Collections");
                                    if ui
                                        .add(
                                            egui::TextEdit::singleline(&mut self.collections_input)
                                                .hint_text("drum rack, live set"),
                                        )
                                        .changed()
                                    {
                                        self.commit_selected_text_fields();
                                    }

                                    ui.separator();
                                    ui.label("Waveform");
                                    draw_waveform(
                                        ui,
                                        &self.waveform_cache,
                                        &self.transient_times,
                                        self.trim_start,
                                        self.trim_end,
                                        self.selected_clip_duration(),
                                    );

                                    ui.label(format!(
                                        "Trim {:.2}s to {:.2}s",
                                        self.trim_start * self.selected_clip_duration(),
                                        self.trim_end * self.selected_clip_duration()
                                    ));
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.trim_start,
                                            0.0..=self.trim_end,
                                        )
                                        .text("Start"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut self.trim_end,
                                            self.trim_start..=1.0,
                                        )
                                        .text("End"),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut self.fade_in_ms, 0.0..=500.0)
                                            .text("Fade in ms"),
                                    );
                                    ui.add(
                                        egui::Slider::new(&mut self.fade_out_ms, 0.0..=500.0)
                                            .text("Fade out ms"),
                                    );
                                    ui.checkbox(
                                        &mut self.normalize_export,
                                        "Normalize preview/export",
                                    );

                                    ui.horizontal_wrapped(|ui| {
                                        if ui.button("Play Source").clicked() {
                                            let path = node.path.clone();
                                            if let Some(clip) = self.selected_clip.clone() {
                                                self.play_clip_region(&path, &clip, false);
                                            }
                                        }
                                        if ui.button("Preview Region").clicked() {
                                            let path = node.path.clone();
                                            if let Some(clip) = self.selected_clip.clone() {
                                                self.play_clip_region(&path, &clip, true);
                                            }
                                        }
                                        if ui.button("Snap To Transients").clicked() {
                                            if let Some((start, end)) = snap_region_to_transients(
                                                self.trim_start,
                                                self.trim_end,
                                                &self.transient_times,
                                                self.selected_clip_duration(),
                                            ) {
                                                self.trim_start = start;
                                                self.trim_end = end;
                                            }
                                        }
                                    });

                                    ui.separator();
                                    ui.label("Export preset");
                                    egui::ComboBox::from_id_source("export_preset")
                                        .selected_text(self.export_preset.label())
                                        .show_ui(ui, |ui| {
                                            for preset in [
                                                ExportPreset::SourceMono16,
                                                ExportPreset::Dawk48Mono24,
                                                ExportPreset::DigitaktMono16,
                                                ExportPreset::Sp404Mono16,
                                                ExportPreset::MpcMono16,
                                            ] {
                                                ui.selectable_value(
                                                    &mut self.export_preset,
                                                    preset,
                                                    preset.label(),
                                                );
                                            }
                                        });
                                    ui.horizontal_wrapped(|ui| {
                                        if ui.button("Export Selected").clicked() {
                                            self.export_selected();
                                        }
                                        if ui.button("Export Filtered Batch").clicked() {
                                            self.export_batch(false);
                                        }
                                        if ui.button("Export Favorites").clicked() {
                                            self.export_batch(true);
                                        }
                                    });
                                });
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
                        self.select_node(idx, true, false);
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
            [
                Pos2::new(screen_x, rect.top()),
                Pos2::new(screen_x, rect.bottom()),
            ],
            Stroke::new(1.0, grid),
        );
        x += step_world;
    }
    let mut y = (top_left.y / step_world).floor() * step_world;
    while y <= bottom_right.y {
        let screen_y = world_to_screen(rect, Vec2::new(center.x, y), center, zoom).y;
        painter.line_segment(
            [
                Pos2::new(rect.left(), screen_y),
                Pos2::new(rect.right(), screen_y),
            ],
            Stroke::new(1.0, grid),
        );
        y += step_world;
    }

    let axis = Color32::from_gray(74);
    if top_left.x <= 0.0 && bottom_right.x >= 0.0 {
        let screen_x = world_to_screen(rect, Vec2::new(0.0, center.y), center, zoom).x;
        painter.line_segment(
            [
                Pos2::new(screen_x, rect.top()),
                Pos2::new(screen_x, rect.bottom()),
            ],
            Stroke::new(1.5, axis),
        );
    }
    if top_left.y <= 0.0 && bottom_right.y >= 0.0 {
        let screen_y = world_to_screen(rect, Vec2::new(center.x, 0.0), center, zoom).y;
        painter.line_segment(
            [
                Pos2::new(rect.left(), screen_y),
                Pos2::new(rect.right(), screen_y),
            ],
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
        let radius = if Some(i) == selected {
            9.5_f32
        } else {
            6.5_f32
        };
        if Some(i) == selected {
            painter.circle_filled(p, radius + 3.0, Color32::from_white_alpha(26));
        }
        painter.circle_filled(p, radius, node.color);
        painter.circle_stroke(p, radius, Stroke::new(1.0, Color32::from_white_alpha(35)));
    }
}

fn draw_waveform(
    ui: &mut egui::Ui,
    waveform: &[(f32, f32)],
    transient_times: &[f32],
    trim_start: f32,
    trim_end: f32,
    duration_s: f32,
) {
    let desired = Vec2::new(ui.available_width(), 120.0);
    let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 8.0, Color32::from_rgb(12, 17, 24));
    painter.rect_stroke(rect, 8.0, Stroke::new(1.0, Color32::from_gray(42)));
    if waveform.is_empty() {
        return;
    }

    let trim_left = rect.left() + rect.width() * trim_start;
    let trim_right = rect.left() + rect.width() * trim_end;
    let selection = Rect::from_min_max(
        Pos2::new(trim_left, rect.top()),
        Pos2::new(trim_right, rect.bottom()),
    );
    painter.rect_filled(
        selection,
        0.0,
        Color32::from_rgba_premultiplied(74, 128, 184, 32),
    );

    let center_y = rect.center().y;
    for (idx, (min_v, max_v)) in waveform.iter().enumerate() {
        let x = rect.left() + rect.width() * (idx as f32 / waveform.len().max(1) as f32);
        let y1 = center_y - max_v * rect.height() * 0.45;
        let y2 = center_y - min_v * rect.height() * 0.45;
        let highlighted = x >= trim_left && x <= trim_right;
        let color = if highlighted {
            Color32::from_rgb(106, 227, 255)
        } else {
            Color32::from_gray(92)
        };
        painter.line_segment(
            [Pos2::new(x, y1), Pos2::new(x, y2)],
            Stroke::new(1.0, color),
        );
    }

    if duration_s > 0.0 {
        for &time in transient_times {
            let t = (time / duration_s).clamp(0.0, 1.0);
            let x = rect.left() + rect.width() * t;
            painter.line_segment(
                [
                    Pos2::new(x, rect.top() + 8.0),
                    Pos2::new(x, rect.bottom() - 8.0),
                ],
                Stroke::new(1.0, Color32::from_rgba_premultiplied(255, 204, 112, 110)),
            );
        }
    }
}

fn snap_region_to_transients(
    trim_start: f32,
    trim_end: f32,
    transient_times: &[f32],
    duration_s: f32,
) -> Option<(f32, f32)> {
    if transient_times.is_empty() || duration_s <= 0.0 {
        return None;
    }
    let start_s = trim_start * duration_s;
    let end_s = trim_end * duration_s;
    let nearest_start = transient_times.iter().copied().min_by(|a, b| {
        (a - start_s)
            .abs()
            .partial_cmp(&(b - start_s).abs())
            .unwrap()
    });
    let nearest_end = transient_times
        .iter()
        .copied()
        .min_by(|a, b| (a - end_s).abs().partial_cmp(&(b - end_s).abs()).unwrap());
    match (nearest_start, nearest_end) {
        (Some(a), Some(b)) if b > a => Some((
            (a / duration_s).clamp(0.0, 1.0),
            (b / duration_s).clamp(0.0, 1.0),
        )),
        _ => None,
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

struct TimbreSummary {
    centroid: f32,
    centroid_std: f32,
    bandwidth: f32,
    rolloff: f32,
    flatness: f32,
    flatness_std: f32,
    attack: f32,
    flux: f32,
    flux_std: f32,
    crest: f32,
    band_stats: Vec<f32>,
    band_balance: [f32; 3],
}

fn feature_node(path: &Path) -> Result<SampleNode> {
    let clip = load_audio_clip(path)?;
    let samples = resample_linear(
        &clip.samples,
        clip.sample_rate,
        44_100,
        Some((44_100.0 * MAX_ANALYSIS_SECONDS) as usize),
    );
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

    let timbre = timbre_features(&samples, 44_100);
    let (transients, bpm, loop_score) = detect_transients_and_bpm(&samples, 44_100);
    let transient_density = (transients.len() as f32 / duration_s.max(0.25)).min(12.0) / 12.0;
    let bpm_norm = bpm.map(|v| ((v - 70.0) / 110.0).clamp(0.0, 1.0)).unwrap_or(0.0);
    let bpm_present = if bpm.is_some() { 1.0 } else { 0.0 };
    let mut vec = vec![
        duration_s / MAX_ANALYSIS_SECONDS,
        rms,
        zcr,
        timbre.centroid / 12_000.0,
        timbre.centroid_std / 6_000.0,
        timbre.bandwidth / 8_000.0,
        timbre.rolloff / 18_000.0,
        timbre.flatness,
        timbre.flatness_std,
        timbre.attack,
        timbre.flux,
        timbre.flux_std,
        timbre.crest / 20.0,
        transient_density,
        bpm_norm,
        bpm_present,
        loop_score,
        timbre.band_balance[0],
        timbre.band_balance[1],
        timbre.band_balance[2],
    ];
    vec.extend(timbre.band_stats.iter().copied());

    Ok(SampleNode {
        path: path.to_path_buf(),
        directory_group: String::new(),
        duration_s,
        sample_rate: clip.sample_rate,
        channels: clip.channels,
        rms,
        centroid: timbre.centroid,
        zcr,
        flatness: timbre.flatness,
        attack: timbre.attack,
        bpm,
        transient_count: transients.len(),
        loop_score,
        vec,
        pos: Vec2::ZERO,
        vel: Vec2::ZERO,
        anchor: Vec2::new(0.5, 0.5),
        color: Color32::from_rgb(110, 227, 255),
        timbre: TimbreClass::Texture,
    })
}

fn timbre_features(samples: &[f32], sr: u32) -> TimbreSummary {
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
    let mut bandwidths = Vec::new();
    let mut rolloffs = Vec::new();
    let mut flatnesses = Vec::new();
    let mut fluxes = Vec::new();
    let mut crests = Vec::new();
    let mut attacks = Vec::new();
    let mut band_accum = vec![0.0f32; edges.len() - 1];
    let mut band_energy_accum = vec![0.0f32; edges.len() - 1];
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
        let freqs: Vec<f32> = (0..(nfft / 2))
            .map(|i| i as f32 * sr as f32 / nfft as f32)
            .collect();
        let sum_mag = mags.iter().sum::<f32>().max(1e-6);
        let centroid = freqs
            .iter()
            .zip(mags.iter())
            .map(|(f, m)| f * m)
            .sum::<f32>()
            / sum_mag;
        let bandwidth = (freqs
            .iter()
            .zip(mags.iter())
            .map(|(f, m)| {
                let delta = *f - centroid;
                delta * delta * *m
            })
            .sum::<f32>()
            / sum_mag)
            .sqrt();
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
        let geo_mag =
            (mags.iter().map(|m| m.max(1e-12).ln()).sum::<f32>() / mags.len().max(1) as f32).exp();
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
            band_energy_accum[band] += sum;
            band_accum[band] += if count > 0.0 {
                (sum / count).ln_1p()
            } else {
                0.0
            };
        }

        centroids.push(centroid);
        bandwidths.push(bandwidth);
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
        return TimbreSummary {
            centroid: 0.0,
            centroid_std: 0.0,
            bandwidth: 0.0,
            rolloff: 0.0,
            flatness: 0.0,
            flatness_std: 0.0,
            attack: 0.0,
            flux: 0.0,
            flux_std: 0.0,
            crest: 0.0,
            band_stats: vec![0.0; edges.len() - 1],
            band_balance: [0.0, 0.0, 0.0],
        };
    }

    let band_stats = band_accum
        .into_iter()
        .map(|v| v / frame_count)
        .collect::<Vec<_>>();
    let total_band_energy = band_energy_accum.iter().sum::<f32>().max(1e-6);
    let low_energy = band_energy_accum[..3].iter().sum::<f32>() / total_band_energy;
    let mid_energy = band_energy_accum[3..5].iter().sum::<f32>() / total_band_energy;
    let high_energy = band_energy_accum[5..].iter().sum::<f32>() / total_band_energy;
    TimbreSummary {
        centroid: mean(&centroids),
        centroid_std: stddev(&centroids),
        bandwidth: mean(&bandwidths),
        rolloff: mean(&rolloffs),
        flatness: mean(&flatnesses),
        flatness_std: stddev(&flatnesses),
        attack: mean(&attacks),
        flux: mean(&fluxes),
        flux_std: stddev(&fluxes),
        crest: mean(&crests),
        band_stats,
        band_balance: [low_energy, mid_energy, high_energy],
    }
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

fn stddev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let avg = mean(xs);
    let var = xs
        .iter()
        .map(|x| {
            let d = *x - avg;
            d * d
        })
        .sum::<f32>()
        / xs.len() as f32;
    var.sqrt()
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
    let vocal_score = (1.2 - (centroid_n - 0.42).abs() * 2.2)
        + (0.30 - node.zcr).max(0.0)
        + (0.45 - node.flatness).max(0.0);
    let drum_score =
        node.attack * 12.0 + node.zcr * 4.0 + node.flatness * 2.5 + (node.centroid / 7000.0);
    let tonal_score = (0.55 - node.flatness).max(0.0) * 3.5
        + (0.25 - node.zcr).max(0.0) * 4.0
        + (0.9 - node.attack * 5.0).max(0.0);
    let texture_score =
        node.flatness * 3.0 + (node.centroid / 8000.0) + (node.attack * 2.0).min(1.5);

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
    let force_scale = force.clamp(0.1, 6.0);
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

fn load_audio_clip(path: &Path) -> Result<AudioClip> {
    if let Some(cached) = AUDIO_CACHE.read().get(path).cloned() {
        return Ok(cached);
    }

    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let clip = if ext == "wav" {
        read_wav_clip(path)?
    } else {
        read_decoder_clip(path)?
    };

    let mut cache = AUDIO_CACHE.write();
    cache.insert(path.to_path_buf(), clip.clone());
    Ok(clip)
}

fn read_wav_clip(path: &Path) -> Result<AudioClip> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().filter_map(|s| s.ok()).collect()
        }
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

    Ok(build_audio_clip(samples, sample_rate, spec.channels.max(1)))
}

fn read_decoder_clip(path: &Path) -> Result<AudioClip> {
    let file = File::open(path)?;
    let decoder = rodio::Decoder::new(BufReader::new(file))?;
    let sample_rate = decoder.sample_rate();
    let channels = decoder.channels();
    let samples = decoder.convert_samples::<f32>().collect::<Vec<_>>();
    Ok(build_audio_clip(samples, sample_rate, channels))
}

fn build_audio_clip(samples: Vec<f32>, sample_rate: u32, channels: u16) -> AudioClip {
    let channels_usize = channels.max(1) as usize;
    let mono = if channels_usize == 1 {
        samples
    } else {
        samples
            .chunks(channels_usize)
            .map(|chunk| chunk.iter().copied().sum::<f32>() / channels_usize as f32)
            .collect::<Vec<_>>()
    };
    let peak = mono.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let duration_s = mono.len() as f32 / sample_rate.max(1) as f32;
    AudioClip {
        samples: mono,
        sample_rate,
        channels,
        duration_s,
        peak,
    }
}

fn build_waveform_overview(samples: &[f32], bins: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || bins == 0 {
        return Vec::new();
    }
    let chunk_size = (samples.len() / bins.max(1)).max(1);
    samples
        .chunks(chunk_size)
        .take(bins)
        .map(|chunk| {
            let mut min_v = 1.0f32;
            let mut max_v = -1.0f32;
            for &sample in chunk {
                min_v = min_v.min(sample);
                max_v = max_v.max(sample);
            }
            (min_v, max_v)
        })
        .collect()
}

fn resample_linear(
    samples: &[f32],
    src_rate: u32,
    target_rate: u32,
    max_len: Option<usize>,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if src_rate == target_rate {
        let mut out = samples.to_vec();
        if let Some(limit) = max_len {
            out.truncate(limit);
        }
        return out;
    }
    let ratio = target_rate as f32 / src_rate.max(1) as f32;
    let new_len = ((samples.len() as f32) * ratio).round().max(1.0) as usize;
    let out_len = max_len.map(|limit| limit.min(new_len)).unwrap_or(new_len);
    let mut res = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f32 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f32;
        let s0 = *samples.get(idx).unwrap_or(&0.0);
        let s1 = *samples.get(idx + 1).unwrap_or(&s0);
        res.push(s0 * (1.0 - frac) + s1 * frac);
    }
    res
}

fn detect_transients_and_bpm(samples: &[f32], sr: u32) -> (Vec<f32>, Option<f32>, f32) {
    let frame = 1024usize;
    let hop = 512usize;
    if samples.len() < frame {
        return (Vec::new(), None, 0.0);
    }

    let mut envelope = Vec::new();
    let mut prev_energy = 0.0f32;
    let mut start = 0usize;
    while start + frame <= samples.len() {
        let energy = samples[start..start + frame]
            .iter()
            .map(|s| s.abs())
            .sum::<f32>()
            / frame as f32;
        envelope.push((energy - prev_energy).max(0.0));
        prev_energy = energy;
        start += hop;
    }
    if envelope.is_empty() {
        return (Vec::new(), None, 0.0);
    }

    let mean_env = envelope.iter().sum::<f32>() / envelope.len() as f32;
    let threshold = mean_env * 2.2;
    let mut transient_times = Vec::new();
    for (idx, &value) in envelope.iter().enumerate() {
        let prev = envelope.get(idx.saturating_sub(1)).copied().unwrap_or(0.0);
        let next = envelope.get(idx + 1).copied().unwrap_or(0.0);
        if value > threshold && value >= prev && value >= next {
            transient_times.push(idx as f32 * hop as f32 / sr as f32);
        }
    }

    let mut intervals = transient_times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|dt| *dt > 0.18 && *dt < 1.5)
        .collect::<Vec<_>>();
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let bpm = intervals
        .get(intervals.len().saturating_sub(1) / 2)
        .copied()
        .map(|median_dt| {
            let mut bpm = 60.0 / median_dt.max(1e-3);
            while bpm < 70.0 {
                bpm *= 2.0;
            }
            while bpm > 180.0 {
                bpm *= 0.5;
            }
            bpm
        });

    let loop_score = if intervals.len() >= 2 {
        let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance = intervals
            .iter()
            .map(|dt| {
                let diff = *dt - mean;
                diff * diff
            })
            .sum::<f32>()
            / intervals.len() as f32;
        let stability = 1.0 - (variance.sqrt() / mean.max(1e-3)).clamp(0.0, 1.0);
        let density = (transient_times.len() as f32 / 16.0).clamp(0.0, 1.0);
        (0.75 * stability + 0.25 * density).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (transient_times, bpm, loop_score)
}

fn parse_csv_field(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn processed_audio_for_export(
    clip: &AudioClip,
    trim_start: f32,
    trim_end: f32,
    fade_in_ms: f32,
    fade_out_ms: f32,
    normalize: bool,
    target_rate: u32,
) -> Vec<f32> {
    let len = clip.samples.len();
    if len == 0 {
        return Vec::new();
    }
    let start_idx = ((trim_start.clamp(0.0, 1.0)) * len as f32).floor() as usize;
    let end_idx = ((trim_end.clamp(0.0, 1.0)) * len as f32).ceil() as usize;
    let start_idx = start_idx.min(len.saturating_sub(1));
    let end_idx = end_idx.max(start_idx + 1).min(len);
    let mut out = clip.samples[start_idx..end_idx].to_vec();

    let fade_in_samples = ((fade_in_ms / 1000.0) * clip.sample_rate as f32) as usize;
    let fade_out_samples = ((fade_out_ms / 1000.0) * clip.sample_rate as f32) as usize;
    for i in 0..fade_in_samples.min(out.len()) {
        let gain = i as f32 / fade_in_samples.max(1) as f32;
        out[i] *= gain;
    }
    for i in 0..fade_out_samples.min(out.len()) {
        let idx = out.len() - 1 - i;
        let gain = i as f32 / fade_out_samples.max(1) as f32;
        out[idx] *= gain;
    }

    if normalize {
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for sample in &mut out {
            *sample /= peak;
        }
    }

    resample_linear(&out, clip.sample_rate, target_rate, None)
}

fn export_clip_to_wav(
    clip: &AudioClip,
    path: &Path,
    trim_start: f32,
    trim_end: f32,
    fade_in_ms: f32,
    fade_out_ms: f32,
    normalize: bool,
    preset: ExportPreset,
) -> Result<()> {
    let sample_rate = preset.sample_rate(clip.sample_rate);
    let samples = processed_audio_for_export(
        clip,
        trim_start,
        trim_end,
        fade_in_ms,
        fade_out_ms,
        normalize,
        sample_rate,
    );
    if samples.is_empty() {
        return Err(anyhow!("empty render"));
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: preset.bits_per_sample(),
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    match preset.bits_per_sample() {
        24 => {
            for sample in samples {
                writer.write_sample((sample.clamp(-1.0, 1.0) * 8_388_607.0) as i32)?;
            }
        }
        _ => {
            for sample in samples {
                writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
            }
        }
    }
    writer.finalize()?;
    Ok(())
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
