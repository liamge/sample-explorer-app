# Sample Explorer App

A basic Python desktop app for sample exploration and lightweight sample creation on macOS.

## Features

- Browse a folder of samples
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

## Supported Formats

- WAV
- AIFF / AIF
- FLAC
- OGG

## Requirements

- Python 3.10+
- macOS recommended

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sample_explorer_app.py
```

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
