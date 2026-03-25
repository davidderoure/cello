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

Usage::

    python generate_sfz.py OUTPUT_DIR [options]

    # Generate SFZ files for all articulations found in the index
    python generate_sfz.py samples/mono_1/

    # Legato only
    python generate_sfz.py samples/mono_1/ --articulations legato

    # Cap the key range extension (default 6 semitones either side)
    python generate_sfz.py samples/mono_1/ --max-range 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_ARTICULATIONS = ["legato", "staccato", "vibrato", "pizzicato"]

# Default maximum semitones a zone extends beyond the outermost sample.
DEFAULT_MAX_RANGE = 6


def name_to_midi(name: str) -> int | None:
    """Convert a note name such as 'A3' or 'C#4' to a MIDI number.

    Uses standard scientific pitch notation (A4 = MIDI 69 = 440 Hz).
    """
    if not name:
        return None
    name = name.strip()
    i = len(name) - 1
    while i > 0 and (name[i].isdigit() or name[i] == "-"):
        i -= 1
    pitch  = name[:i + 1].upper().replace("B", "A#")   # normalise flats
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
) -> Path:
    """Write one SFZ file for a single articulation.

    Args:
        art:        Articulation name (used for filename and comments).
        notes:      Mapping of midi_note → [wav_path, ...] (sorted by take).
        output_dir: Root output directory (SFZ written here; paths are relative).
        max_range:  Maximum key-range extension beyond the outermost sample.

    Returns:
        Path to the written SFZ file.
    """
    ranges = key_ranges(list(notes.keys()), max_range=max_range)
    lines  = [
        f"// {art.capitalize()} — generated by generate_sfz.py",
        f"// {len(notes)} notes, "
        f"{sum(len(v) for v in notes.values())} total samples",
        f"// Load this file in sfizz (https://sfizz.com) as an Audio Unit in Logic Pro.",
        "",
    ]

    for midi in sorted(notes.keys()):
        wav_paths = notes[midi]
        lo, hi    = ranges[midi]
        n_takes   = len(wav_paths)
        note_name = midi_to_sfz_name(midi)

        lines.append(f"// ── {note_name}  ({n_takes} take{'s' if n_takes > 1 else ''}) ──")

        for seq_pos, wav_path in enumerate(wav_paths, start=1):
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    samples = load_samples(args.output_dir, args.articulations)

    if not samples:
        print("No samples found. Check that _index.csv exists and contains "
              "the requested articulations.", file=sys.stderr)
        return 1

    for art, notes in samples.items():
        if not notes:
            print(f"  {art:12s}  no samples — skipped")
            continue

        sfz_path = generate_sfz(art, notes, args.output_dir, args.max_range)
        n_notes  = len(notes)
        n_takes  = sum(len(v) for v in notes.values())
        print(f"  {art:12s}  {n_notes} notes  {n_takes} samples  →  {sfz_path.name}")

    print(f"\nSFZ files written to {args.output_dir}")
    print("\nTo use in Logic Pro:")
    print("  1. Install sfizz Audio Unit: https://sfizz.com/downloads")
    print("  2. Create a Software Instrument track in Logic")
    print("  3. Load sfizz as the instrument plugin")
    print("  4. In sfizz, click the file icon and open the .sfz file")

    return 0


if __name__ == "__main__":
    sys.exit(main())
