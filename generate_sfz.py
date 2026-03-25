"""generate_sfz.py — create SFZ instrument files from a process_recording output directory.

SFZ is a plain-text, open standard for sample libraries supported by many
free and commercial players.  On macOS / Logic Pro the recommended free
player is **sfizz** (https://sfizz.com), available as an Audio Unit plugin.

One SFZ file is written per articulation (legato.sfz, staccato.sfz, etc.)
into the root output directory alongside the per-articulation sub-folders.

Zone layout
-----------
* Each note with at least one sample gets its own zone.
* The key range for each zone extends halfway to the nearest neighbour on
  each side, so the keyboard is covered with no gaps and no overlaps.
* Multiple takes of the same note are grouped with ``seq_position`` for
  round-robin playback — each keypress cycles to the next take.

Loop mode
---------
The default loop mode is ``no_loop``: each sample plays once in full, then
stops.  This avoids splice artefacts entirely and works well for legato
cello where the natural sustain (typically 3–8 s) is long enough for most
musical contexts.

To use sustain looping instead (requires loop markers embedded in the WAV
files by ``add_loop_points.py``) pass ``--loop-mode loop_sustain``.  In
``loop_sustain`` mode sfizz loops through the sustain section while the key
is held and plays the natural tail when the key is released.

Volume normalisation
--------------------
All takes of the same note are normalised to the loudest take using the
SFZ ``volume`` attribute (a dB offset applied per region).  This keeps
round-robin takes at a consistent level while preserving natural dynamics
between different notes.  WAV files are never modified.

Usage::

    python generate_sfz.py OUTPUT_DIR [options]

    # Generate SFZ files for all articulations found in the index
    python generate_sfz.py samples/mono_1/

    # Legato only
    python generate_sfz.py samples/mono_1/ --articulations legato

    # Cap the key range extension (default 6 semitones either side)
    python generate_sfz.py samples/mono_1/ --max-range 4

    # Use sustain looping (requires loop markers embedded by add_loop_points.py)
    python generate_sfz.py samples/mono_1/ --loop-mode loop_sustain

    # Skip volume normalisation
    python generate_sfz.py samples/mono_1/ --no-normalize
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_ARTICULATIONS = ["legato", "staccato", "vibrato", "pizzicato"]

# Default maximum semitones a zone extends beyond the outermost sample.
DEFAULT_MAX_RANGE = 6

# SFZ loop modes understood by sfizz.
VALID_LOOP_MODES = ("no_loop", "loop_continuous", "loop_sustain", "one_shot")
DEFAULT_LOOP_MODE = "no_loop"

# Enharmonic flat → sharp equivalents (after upper-casing the pitch token).
# "BB" is Bb, not the note B — standalone "B" is left unchanged.
_FLAT_TO_SHARP: dict[str, str] = {
    "BB": "A#",
    "EB": "D#",
    "AB": "G#",
    "DB": "C#",
    "GB": "F#",
    "CB": "B",
    "FB": "E",
}


def name_to_midi(name: str) -> int | None:
    """Convert a note name such as 'A3', 'C#4', or 'Bb3' to a MIDI number.

    Uses standard scientific pitch notation (A4 = MIDI 69 = 440 Hz).
    Flat suffixes (e.g. 'Bb', 'Eb') are normalised to their sharp equivalents.
    """
    if not name:
        return None
    name = name.strip()
    i = len(name) - 1
    while i > 0 and (name[i].isdigit() or name[i] == "-"):
        i -= 1
    pitch  = name[:i + 1].upper()
    pitch  = _FLAT_TO_SHARP.get(pitch, pitch)   # normalise flats; leaves "B" intact
    octave = name[i + 1:]
    if pitch not in NOTE_NAMES or not octave.lstrip("-").isdigit():
        return None
    return (int(octave) + 1) * 12 + NOTE_NAMES.index(pitch)


def midi_to_sfz_name(midi: int) -> str:
    """Return the SFZ note name string for a MIDI note number (e.g. 69 → 'A4')."""
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


# ---------------------------------------------------------------------------
# Load and organise the index
# ---------------------------------------------------------------------------


def load_samples(
    output_dir: Path,
    articulations: list[str],
) -> dict[str, dict[int, list[Path]]]:
    """Return samples organised by articulation → midi_note → [file, ...].

    Args:
        output_dir:    Root output directory from process_recording.py.
        articulations: Articulation types to include.

    Returns:
        Nested dict: articulation → midi_note → ordered list of WAV paths.
    """
    index_path = output_dir / "_index.csv"
    if not index_path.exists():
        print(f"Error: _index.csv not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    # art → midi → [(take, path)]
    raw: dict[str, dict[int, list[tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            art  = row.get("articulation", "").strip()
            note = row.get("note", "").strip()
            file = row.get("file", "").strip()
            take = int(row.get("take", 1))

            if art not in articulations or not note or not file:
                continue

            midi = name_to_midi(note)
            if midi is None:
                continue

            wav_path = output_dir / file
            raw[art][midi].append((take, wav_path))

    # Sort each note's list by take number.
    result: dict[str, dict[int, list[Path]]] = {}
    for art, notes in raw.items():
        result[art] = {
            midi: [p for _, p in sorted(takes)]
            for midi, takes in notes.items()
        }

    return result


# ---------------------------------------------------------------------------
# Volume normalisation helpers
# ---------------------------------------------------------------------------

_SOUNDFILE_AVAILABLE: bool | None = None  # lazily resolved


def _soundfile_available() -> bool:
    global _SOUNDFILE_AVAILABLE
    if _SOUNDFILE_AVAILABLE is None:
        try:
            import soundfile  # noqa: F401
            _SOUNDFILE_AVAILABLE = True
        except ImportError:
            _SOUNDFILE_AVAILABLE = False
    return _SOUNDFILE_AVAILABLE


def peak_db(wav_path: Path) -> float | None:
    """Return the peak amplitude of *wav_path* in dBFS, or None on error.

    Reads all channels; the peak is the maximum absolute sample value across
    all channels.  Returns ``None`` if soundfile is unavailable or the file
    cannot be read.
    """
    if not _soundfile_available():
        return None
    try:
        import soundfile as sf
        import numpy as np
        data, _ = sf.read(str(wav_path), dtype="float32", always_2d=True)
        peak = float(np.max(np.abs(data)))
        if peak <= 0.0:
            return None
        return 20.0 * math.log10(peak)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s for normalisation: %s", wav_path, exc)
        return None


def note_volume_offsets(wav_paths: list[Path]) -> list[float]:
    """Return per-take dB offsets that bring all takes to the level of the loudest.

    The loudest take gets offset 0.0 dB; quieter takes get a positive offset
    so they are boosted to match.  Returns a list of zeros if soundfile is
    unavailable or no peaks could be measured.

    Args:
        wav_paths: Ordered list of WAV paths for one note (all takes).

    Returns:
        List of dB offsets, one per take, same order as *wav_paths*.
    """
    peaks = [peak_db(p) for p in wav_paths]
    valid  = [db for db in peaks if db is not None]
    if not valid:
        return [0.0] * len(wav_paths)

    max_db = max(valid)
    offsets: list[float] = []
    for db in peaks:
        if db is None:
            offsets.append(0.0)
        else:
            offsets.append(max_db - db)   # always >= 0
    return offsets


# ---------------------------------------------------------------------------
# Key-range calculation
# ---------------------------------------------------------------------------


def key_ranges(
    midi_notes: list[int],
    max_range: int = DEFAULT_MAX_RANGE,
) -> dict[int, tuple[int, int]]:
    """Calculate (lo_key, hi_key) for each note.

    Splits the keyboard midway between adjacent samples.  The outermost
    samples extend by at most *max_range* semitones.

    Args:
        midi_notes: Sorted list of MIDI note numbers that have samples.
        max_range:  Maximum semitones to extend beyond the outermost note.

    Returns:
        Dict mapping each MIDI note to its (lo_key, hi_key) inclusive range.
    """
    notes  = sorted(midi_notes)
    ranges = {}
    for i, note in enumerate(notes):
        lo = (notes[i - 1] + note) // 2 + 1 if i > 0 else max(0, note - max_range)
        hi = (note + notes[i + 1]) // 2 if i < len(notes) - 1 else min(127, note + max_range)
        ranges[note] = (lo, hi)
    return ranges


# ---------------------------------------------------------------------------
# SFZ generation
# ---------------------------------------------------------------------------


def generate_sfz(
    art: str,
    notes: dict[int, list[Path]],
    output_dir: Path,
    max_range: int,
    normalize: bool = True,
    loop_mode: str = DEFAULT_LOOP_MODE,
) -> Path:
    """Write one SFZ file for a single articulation.

    Args:
        art:        Articulation name (used for filename and comments).
        notes:      Mapping of midi_note → [wav_path, ...] (sorted by take).
        output_dir: Root output directory (SFZ written here; paths are relative).
        max_range:  Maximum key-range extension beyond the outermost sample.
        normalize:  If True, add per-note ``volume`` offsets so all round-robin
                    takes of the same note play at the same peak level.
        loop_mode:  SFZ loop_mode value written into every region.  Default is
                    ``no_loop`` — samples play once in full with no splicing.
                    Use ``loop_sustain`` if WAV files contain embedded loop
                    markers (added by ``add_loop_points.py``).

    Returns:
        Path to the written SFZ file.
    """
    ranges = key_ranges(list(notes.keys()), max_range=max_range)

    norm_note = "(volume-normalised)" if normalize else "(no normalisation)"
    lines  = [
        f"// {art.capitalize()} — generated by generate_sfz.py",
        f"// {len(notes)} notes, "
        f"{sum(len(v) for v in notes.values())} total samples  {norm_note}",
        f"// loop_mode={loop_mode}",
        f"// Load this file in sfizz (https://sfizz.com) as an Audio Unit in Logic Pro.",
        "",
        f"<global> loop_mode={loop_mode}",
        "",
    ]

    if normalize and not _soundfile_available():
        logger.warning(
            "soundfile not installed — volume normalisation skipped. "
            "Run: pip install soundfile"
        )
        normalize = False

    for midi in sorted(notes.keys()):
        wav_paths = notes[midi]
        lo, hi    = ranges[midi]
        n_takes   = len(wav_paths)
        note_name = midi_to_sfz_name(midi)

        # Compute per-take volume offsets (all zeros when normalize=False).
        offsets = note_volume_offsets(wav_paths) if normalize else [0.0] * n_takes

        lines.append(f"// ── {note_name}  ({n_takes} take{'s' if n_takes > 1 else ''}) ──")

        for seq_pos, (wav_path, vol_offset) in enumerate(
            zip(wav_paths, offsets), start=1
        ):
            # Path relative to the SFZ file (both live in output_dir).
            rel = wav_path.relative_to(output_dir)

            if n_takes > 1:
                lines.append(f"<group> seq_length={n_takes} seq_position={seq_pos}")

            lines.append("<region>")
            lines.append(f"  sample={rel.as_posix()}")
            lines.append(f"  pitch_keycenter={midi}")
            lines.append(f"  lokey={lo}")
            lines.append(f"  hikey={hi}")
            lines.append(f"  lovel=0  hivel=127")
            if normalize and vol_offset != 0.0:
                lines.append(f"  volume={vol_offset:.2f}")
            lines.append("")

    sfz_path = output_dir / f"{art}.sfz"
    sfz_path.write_text("\n".join(lines), encoding="utf-8")
    return sfz_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_sfz",
        description="Generate SFZ instrument files from a process_recording output directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Root output directory produced by process_recording.py.",
    )
    p.add_argument(
        "--articulations",
        nargs="+",
        default=ALL_ARTICULATIONS,
        choices=ALL_ARTICULATIONS,
        metavar="ART",
        help="Articulations to generate (default: all four).",
    )
    p.add_argument(
        "--max-range",
        type=int,
        default=DEFAULT_MAX_RANGE,
        metavar="SEMITONES",
        help="Max semitones a zone extends beyond the outermost sample.",
    )
    p.add_argument(
        "--loop-mode",
        default=DEFAULT_LOOP_MODE,
        choices=VALID_LOOP_MODES,
        metavar="MODE",
        help=(
            "SFZ loop_mode for every region "
            f"({', '.join(VALID_LOOP_MODES)}). "
            "Default 'no_loop' plays each sample once in full — "
            "avoids splice artefacts and suits natural legato sustains. "
            "Use 'loop_sustain' if WAV files have embedded loop markers "
            "(added by add_loop_points.py)."
        ),
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Disable per-note volume normalisation (useful if soundfile is not installed).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.WARNING)

    args = build_parser().parse_args(argv)
    normalize = not args.no_normalize
    loop_mode = args.loop_mode

    samples = load_samples(args.output_dir, args.articulations)

    if not samples:
        print("No samples found. Check that _index.csv exists and contains "
              "the requested articulations.", file=sys.stderr)
        return 1

    if normalize and not _soundfile_available():
        print(
            "Warning: soundfile not installed — volume normalisation skipped.\n"
            "         Run: pip install soundfile",
            file=sys.stderr,
        )
        normalize = False

    for art, notes in samples.items():
        if not notes:
            print(f"  {art:12s}  no samples — skipped")
            continue

        sfz_path = generate_sfz(
            art, notes, args.output_dir, args.max_range,
            normalize=normalize, loop_mode=loop_mode,
        )
        n_notes  = len(notes)
        n_takes  = sum(len(v) for v in notes.values())
        norm_tag = "" if normalize else "  (no normalisation)"
        print(f"  {art:12s}  {n_notes} notes  {n_takes} samples  →  {sfz_path.name}{norm_tag}")

    print(f"\nSFZ files written to {args.output_dir}")
    print("\nTo use in Logic Pro:")
    print("  1. Install sfizz Audio Unit: https://sfizz.com/downloads")
    print("  2. Create a Software Instrument track in Logic")
    print("  3. Load sfizz as the instrument plugin")
    print("  4. In sfizz, click the file icon and open the .sfz file")

    return 0


if __name__ == "__main__":
    sys.exit(main())
