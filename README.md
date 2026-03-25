# cello-sampler

Automated pipeline for extracting and classifying individual note samples from multi-channel studio recordings of solo cello. Designed for building DAW sample libraries from hours of session material.

## What it does

1. **Streams** large audio files in 10-second chunks — handles multi-hour recordings without loading everything into RAM
2. **Detects note onsets** using a Superflux spectral-flux envelope, band-limited to the cello frequency range (60–4000 Hz), with an adaptive threshold and non-maximum suppression
3. **Rejects polyphonic samples** using a multi-pitch salience map (harmonics 1–8) and Harmonic Product Spectrum cross-check
4. **Rejects poor intonation** using [CREPE](https://github.com/marl/crepe) neural pitch estimation — more robust than autocorrelation or YIN on cello due to the instrument's variable harmonic balance under bow pressure
5. **Classifies articulation** from the amplitude envelope and pitch contour:
   - **Pizzicato** — fast attack (< 15 ms), rapid decay (> 0.5 dB/ms), short duration (< 400 ms)
   - **Staccato** — short total duration (< 250 ms)
   - **Vibrato** — pitch modulation 4–8 Hz with depth > 20 cents; minimum 750 ms duration (≥ 3 oscillation cycles)
   - **Legato** — everything else; minimum 500 ms duration
6. **Writes labelled WAV files** preserving all source channels at the original sample rate, plus a CSV index for DAW import
7. **Embeds loop points** into legato and vibrato samples via `add_loop_points.py`, so Logic Pro Sampler (and other samplers) can sustain notes indefinitely

## Requirements

- Python 3.10+
- [libsndfile](https://libsndfile.github.io/libsndfile/) (required by `soundfile` — `brew install libsndfile` on macOS)
- TensorFlow (required by `crepe` — install separately, see [CREPE docs](https://github.com/marl/crepe))

```bash
pip install -r requirements.txt
```

### Installing crepe

On Python 3.12 the standard `pip install crepe` may fail with a `pkg_resources` error. Use:

```bash
pip install --upgrade pip setuptools wheel
pip install crepe --no-build-isolation
```

Or install directly from source:

```bash
pip install git+https://github.com/marl/crepe.git
```

## Usage

### Extract samples

```bash
python process_recording.py INPUT_FILE OUTPUT_DIR [options]
```

Example:

```bash
python process_recording.py session_day1.wav ~/samples/cello/
```

### Add loop points (legato and vibrato)

Run after `process_recording.py` to embed sustain loop markers into WAV files.
Logic Pro Sampler reads these automatically.

```bash
python add_loop_points.py ~/samples/cello/

# Preview without modifying files
python add_loop_points.py ~/samples/cello/ --dry-run -v

# Legato only, with looser splice tolerance
python add_loop_points.py ~/samples/cello/ --articulation legato --max-splice-error 0.05
```

### Output structure

```
~/samples/cello/
├── legato/
│   ├── A3_legato_001.wav
│   ├── A3_legato_002.wav
│   └── ...
├── staccato/
│   └── D4_staccato_001.wav
├── vibrato/
│   └── A3_vibrato_001.wav
├── pizzicato/
│   └── G3_pizzicato_001.wav
├── rejected/
│   ├── polyphonic/       <- JSON sidecars with diagnostic detail
│   ├── intonation/
│   ├── low_confidence/
│   └── too_short/
└── _index.csv            <- one row per accepted note, importable into most DAWs
```

Filenames follow the convention `{note}{octave}_{articulation}_{take:03d}.wav`
(e.g. `A3_legato_002.wav`).

### All options

```
positional arguments:
  input_file            Path to source audio file (WAV, AIFF, FLAC; 48 kHz recommended)
  output_dir            Root directory for labelled output files. Created if absent.

I/O options:
  --workers N           Parallel worker processes (default: 4)
  --chunk-seconds SECS  Streaming chunk size in seconds (default: 10.0)
  --overlap-seconds SECS  Carry-buffer overlap between chunks (default: 2.0)

Onset detection:
  --onset-threshold MULT  Onset strength as multiple of local mean (default: 1.5)
  --min-note-ms MS        Minimum inter-onset gap in ms (default: 50.0)

Quality gating:
  --pitch-confidence CONF       Minimum CREPE confidence 0-1 (default: 0.85)
  --max-intonation-cents CENTS  Max deviation from 12-TET in cents (default: 30.0)
  --polyphony-threshold FRAC    Secondary pitch salience fraction 0-1 (default: 0.30)

Articulation classification:
  --pizz-max-attack-ms MS         Max attack duration for pizzicato (default: 15.0)
  --stacc-max-duration-ms MS      Max duration for staccato (default: 250.0)
  --vibrato-min-depth-cents CENTS Min pitch modulation depth for vibrato (default: 20.0)

  -v, --verbose         Enable DEBUG-level logging
```

## Tuning for your session

All thresholds live in `cello_sampler/config.py` and can be overridden from the
command line without touching source code. Common adjustments:

| Symptom | Fix |
|---|---|
| Too many onsets in noisy passages | Raise `--onset-threshold` (e.g. `2.0`) |
| Quiet notes missed | Lower `--onset-threshold` (e.g. `1.2`) |
| Good notes rejected for polyphony | Raise `--polyphony-threshold` (e.g. `0.45`) |
| Legato notes classified as staccato | Raise `--stacc-max-duration-ms` |
| Vibrato not detected | Lower `--vibrato-min-depth-cents` |
| Many notes rejected for low confidence | Lower `--pitch-confidence` (e.g. `0.75`) |
| Too many short legato/vibrato takes rejected | Raise `MIN_LEGATO_DURATION_MS` / `MIN_VIBRATO_DURATION_MS` in `config.py` |
| Loop points found in few files | Raise `--max-splice-error` in `add_loop_points.py` (e.g. `0.05`) |

## Running on Colab (GPU)

`colab_pipeline.ipynb` runs the full pipeline on Google Colab with a T4 GPU,
reading and writing audio files from Google Drive. Open it directly:
[colab.research.google.com](https://colab.research.google.com) -> GitHub ->
`davidderoure/cello`.

## Architecture

```
process_recording.py      entry point
add_loop_points.py        loop-point embedding utility
colab_pipeline.ipynb      Google Colab notebook
cello_sampler/
├── config.py             all tuneable constants
├── models.py             dataclasses (NoteCandidate, ClassifiedNote, ...)
├── io.py                 streaming reader, SampleWriter
├── onset.py              Superflux onset detection, note segmentation
├── polyphony.py          HPS + multi-pitch salience polyphony detection
├── pitch.py              CREPE pitch estimation, intonation gating
├── articulation.py       Hilbert envelope features, rule-based classifier
├── pipeline.py           orchestrator, ProcessPoolExecutor parallelism
└── cli.py                argparse interface
```

### Key design decisions

- **No resampling for output** — audio is read and written at the native 48 kHz sample rate; only the mono mix sent to CREPE is downsampled to 16 kHz
- **Streaming with carry-buffer** — a 2-second overlap between chunks prevents notes at chunk boundaries being split across two segments
- **CREPE over YIN/autocorrelation** — cello's variable harmonic balance under bow pressure causes fundamental estimation failures with classical methods; CREPE treats pitch as a 360-bin classification problem and is robust to this
- **Fundamental energy guard in polyphony detection** — a secondary pitch is only counted as a second voice if spectral energy exists at its own fundamental frequency; this eliminates sub-harmonic artefacts without losing genuine double-stop detection
- **Minimum duration gates** — legato (500 ms) and vibrato (750 ms) notes below threshold are rejected post-classification, since short takes are not useful for looped playback

## Running the tests

```bash
pytest
```

The test suite (94 tests) runs without TensorFlow or a real audio file — CREPE
and resampy are stubbed automatically by `tests/conftest.py`.
