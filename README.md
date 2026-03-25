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
   - **Vibrato** — pitch modulation 4–8 Hz with depth > 8 cents std dev; minimum 500 ms duration
   - **Legato** — everything else; minimum 500 ms duration
6. **Writes labelled WAV files** preserving all source channels at the original sample rate, plus a CSV index for DAW import
7. **Generates SFZ instrument files** via `generate_sfz.py` with explicit MIDI key mappings, `no_loop` playback by default, and per-note volume normalisation for consistent round-robin playback

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

### Add loop points (optional)

Embeds sustain loop markers into legato and vibrato WAV files.

**For legato samples this is usually not worth it.** Legato cello tone evolves
continuously (bow pressure, natural vibrato, resonance) so any splice point
tends to be audible. The default `no_loop` SFZ mode plays each sample once in
full — completely natural, no artefacts. Skip this step unless your recordings
are unusually stable and you specifically need infinite sustain.

If you do run it, use `--loop-mode loop_sustain` in `generate_sfz.py` so the
embedded markers are actually used.

```bash
python add_loop_points.py ~/samples/cello/

# Preview without modifying files
python add_loop_points.py ~/samples/cello/ --dry-run -v

# Legato only, with looser splice tolerance
python add_loop_points.py ~/samples/cello/ --articulation legato --max-splice-error 0.05
```

### Check coverage

Print a report showing which notes have samples and flag gaps that are too wide
to fill by pitch-shifting from a neighbour.

```bash
python check_coverage.py ~/samples/cello/
```

### Generate SFZ files

Produce one `.sfz` instrument file per articulation with explicit MIDI key
zones and per-note volume normalisation. Load in
[sfizz](https://sfizz.com) (free Audio Unit for Logic Pro).

```bash
python generate_sfz.py ~/samples/cello/

# Legato only
python generate_sfz.py ~/samples/cello/ --articulations legato

# Cap the key range extension (default 6 semitones either side)
python generate_sfz.py ~/samples/cello/ --max-range 4

# Use sustain looping (only if you ran add_loop_points.py first)
python generate_sfz.py ~/samples/cello/ --loop-mode loop_sustain

# Skip volume normalisation
python generate_sfz.py ~/samples/cello/ --no-normalize
```

**Loop modes:**

| Mode | Behaviour | When to use |
|---|---|---|
| `no_loop` (default) | Sample plays once in full, then stops | Always — avoids splice artefacts |
| `loop_sustain` | Loops while key held; plays tail on release | Only if loop markers were embedded by `add_loop_points.py` |
| `loop_continuous` | Loops indefinitely, ignores release | Rare; requires very clean loop points |

**Volume normalisation** reads the peak level of each WAV file and adds a `volume`
dB offset in the SFZ so all round-robin takes of the same note play at the
same level. WAV files are never modified. Inter-note dynamics (a soft B5 stays
quieter than a full-bow A2) are preserved.

To use the generated SFZ in Logic Pro:

1. Install sfizz Audio Unit: https://sfizz.com/downloads
2. Create a Software Instrument track in Logic
3. Load sfizz as the instrument plugin
4. In sfizz, click the file icon and open the `.sfz` file

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
├── _index.csv            <- one row per accepted note, importable into most DAWs
├── legato.sfz            <- generated by generate_sfz.py
├── staccato.sfz
├── vibrato.sfz
└── pizzicato.sfz
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
  --vibrato-min-depth-cents CENTS Min pitch modulation depth for vibrato (default: 8.0)

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
| Vibrato not detected | Lower `--vibrato-min-depth-cents` (default is already 8.0) |
| Many notes rejected for low confidence | Lower `--pitch-confidence` (e.g. `0.75`) |
| Too many short legato/vibrato takes rejected | Raise `MIN_LEGATO_DURATION_MS` / `MIN_VIBRATO_DURATION_MS` in `config.py` |
| Audible splice/click in looped playback | Use default `no_loop` mode instead of `loop_sustain` |
| Loop points found in few files | Raise `--max-splice-error` in `add_loop_points.py` (e.g. `0.05`) |
| Round-robin takes differ in volume | Run `generate_sfz.py` with normalisation enabled (default) |

## Running on Colab (GPU)

`colab_pipeline.ipynb` runs the full pipeline on Google Colab with a T4 GPU,
reading and writing audio files from Google Drive. Open it directly:
[colab.research.google.com](https://colab.research.google.com) -> GitHub ->
`davidderoure/cello`.

## Architecture

```
process_recording.py      entry point
add_loop_points.py        loop-point embedding utility
check_coverage.py         note coverage report
generate_sfz.py           SFZ instrument file generator
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
- **Minimum duration gates** — legato (500 ms) and vibrato (500 ms) notes below threshold are rejected post-classification, since very short takes are poor samples regardless of playback mode
- **`no_loop` by default** — legato cello tone evolves continuously (bow pressure, natural vibrato, body resonance), so any fixed splice point produces an audible artefact; the default SFZ mode plays each sample once in full, which is natural and artefact-free; `loop_sustain` is available for users who need infinite sustain and have stable recordings
- **Vibrato depth as std dev** — `VIBRATO_MIN_DEPTH_CENTS` is a standard deviation threshold on the pitch contour, not peak-to-peak; for sinusoidal vibrato, std dev = amplitude/√2, so 8 cents std dev corresponds to roughly ±11 cents (22 cents peak-to-peak), typical for cello
- **SFZ over Logic Sampler auto-map** — Logic Pro's filename-based auto-mapping is unreliable for custom naming conventions; `generate_sfz.py` writes explicit `pitch_keycenter`, `lokey`, and `hikey` values so every note lands on the correct key
- **Non-destructive normalisation** — round-robin takes are level-matched using the SFZ `volume` attribute (a per-region dB offset); WAV files are never modified, preserving the original recordings

## Running the tests

```bash
pytest
```

The test suite (133 tests) runs without TensorFlow or a real audio file — CREPE
and resampy are stubbed automatically by `tests/conftest.py`.
