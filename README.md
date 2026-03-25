# Sample Explorer App

A basic Rust desktop app for sample exploration and lightweight sample creation on macOS.

## Features

- Browse a folder of samples as a 2D cluster projection
- Search sample filenames
- Preview audio files
- View a simple waveform
- Trim start and end points
- Apply fade in / fade out
- Normalize audio
- Reverse audio
- Convert to mono
- Export processed samples
- Export a quick one-shot version
- Graph view that clusters samples by similarity with adjustable force (Obsidian-style)

## Supported Formats

- WAV
- AIFF / AIF
- FLAC
- OGG

## Requirements

- Python 3.10+
- macOS recommended

## Setup (Rust version – default)

```bash
cd sample_explorer_rs
cargo run --release
```

Python UI remains in `sample_explorer_app.py`, but the Rust `eframe`/`rodio` app is the default and fastest way to explore clusters.

## Notes for macOS

- The first run may require permission for audio access depending on your system settings.
- If playback does not work immediately, check your macOS audio output device and permissions.

## Repo Structure

```text
sample-explorer-app/
├── .gitignore
├── README.md
├── requirements.txt
└── sample_explorer_app.py
```

## Ideas for v2

- transient detection and slicing
- BPM / loop-length detection
- spectrogram view
- keyboard shortcuts
- tags / favorites
- drag-and-drop export folders
- pitch shifting and time stretching
- Digitakt-ready export presets
