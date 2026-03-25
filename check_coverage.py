"""check_coverage.py — report which cello notes have samples and which are missing.

Reads the _index.csv produced by process_recording.py and prints a
chromatic map of the full cello range showing:

  ●  note has at least one sample for every requested articulation
  ◑  note has samples for some articulations only
  ○  note has no samples at all (will be pitch-shifted by the sampler)

Also prints a per-articulation table and lists the missing notes so you
know exactly what to re-record.

Usage::

    python check_coverage.py OUTPUT_DIR [options]

    # Check a single mic output
    python check_coverage.py samples/mono_1/

    # Check only legato and vibrato
    python check_coverage.py samples/mono_1/ --articulations legato vibrato

    # Show how many takes exist per note
    python check_coverage.py samples/mono_1/ --takes
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Cello range: C2 (MIDI 36) to B5 (MIDI 83)
# ---------------------------------------------------------------------------

CELLO_MIDI_MIN = 36   # C2 — open C string
CELLO_MIDI_MAX = 83   # B5 — upper practical limit

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ALL_ARTICULATIONS = ["legato", "staccato", "vibrato", "pizzicato"]

# Gap sizes considered acceptable for sampler pitch-shifting.
ACCEPTABLE_GAP = 2    # semitones — shown differently in output


def midi_to_name(midi: int) -> str:
    """Return scientific pitch name for a MIDI note number (e.g. 69 → 'A4')."""
    octave = (midi // 12) - 1
    name   = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------


def load_index(output_dir: Path) -> list[dict]:
    """Read _index.csv from *output_dir* and return all rows as dicts."""
    index_path = output_dir / "_index.csv"
    if not index_path.exists():
        print(f"Error: no _index.csv found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    with index_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(
    rows: list[dict],
    articulations: list[str],
) -> dict[int, dict[str, int]]:
    """Return a mapping of midi_note → {articulation: take_count}.

    Args:
        rows:          All rows from _index.csv.
        articulations: Articulation types to check.

    Returns:
        Dict keyed by MIDI note number; values are dicts of
        articulation → number of takes available.
    """
    coverage: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        art = row.get("articulation", "").strip()
        if art not in articulations:
            continue
        # Derive MIDI note from the note name in the filename or CSV.
        note_name = row.get("note", "").strip()
        midi = name_to_midi(note_name)
        if midi is None:
            continue
        coverage[midi][art] += 1

    return coverage


def name_to_midi(name: str) -> int | None:
    """Convert a note name like 'A3' or 'C#4' to a MIDI number."""
    if not name:
        return None
    # Split off octave digit(s) at the end.
    i = len(name) - 1
    while i > 0 and (name[i].isdigit() or name[i] == "-"):
        i -= 1
    pitch  = name[:i + 1].upper()
    octave = name[i + 1:]
    if pitch not in NOTE_NAMES or not octave.lstrip("-").isdigit():
        return None
    return (int(octave) + 1) * 12 + NOTE_NAMES.index(pitch)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _symbol(covered: int, total: int) -> str:
    if covered == 0:
        return "○"
    if covered == total:
        return "●"
    return "◑"


def print_chromatic_map(
    coverage: dict[int, dict[str, int]],
    articulations: list[str],
) -> None:
    """Print a two-octave-wide chromatic grid of the cello range."""
    n_arts = len(articulations)
    print("\nChromatic coverage map")
    print(f"  ●  all {n_arts} articulation(s) present")
    print( "  ◑  some articulations present")
    print( "  ○  no samples (sampler will pitch-shift from nearest neighbour)\n")

    # Print in rows of 12 (one octave per row).
    for octave_start in range(CELLO_MIDI_MIN, CELLO_MIDI_MAX + 1, 12):
        octave_end = min(octave_start + 11, CELLO_MIDI_MAX)
        notes = range(octave_start, octave_end + 1)

        # Note names header.
        header = "  ".join(f"{midi_to_name(m):4s}" for m in notes)
        print("  " + header)

        # Symbol row.
        symbols = []
        for m in notes:
            arts_present = sum(
                1 for a in articulations if coverage.get(m, {}).get(a, 0) > 0
            )
            sym = _symbol(arts_present, n_arts)
            symbols.append(f"  {sym} ")
        print("  " + " ".join(symbols))
        print()


def print_articulation_table(
    coverage: dict[int, dict[str, int]],
    articulations: list[str],
    show_takes: bool,
) -> None:
    """Print a per-articulation summary table."""
    col_w = 10 if show_takes else 6

    # Header.
    header = f"{'Note':6s}" + "".join(f"{a:{col_w}s}" for a in articulations)
    print(header)
    print("-" * len(header))

    for midi in range(CELLO_MIDI_MIN, CELLO_MIDI_MAX + 1):
        name = midi_to_name(midi)
        row_data = coverage.get(midi, {})
        if not any(row_data.get(a, 0) > 0 for a in articulations):
            continue   # skip completely empty notes

        cols = []
        for art in articulations:
            takes = row_data.get(art, 0)
            if show_takes:
                cols.append(f"{takes:<{col_w}d}" if takes else f"{'—':<{col_w}s}")
            else:
                cols.append(f"{'●':<{col_w}s}" if takes else f"{'○':<{col_w}s}")

        print(f"{name:6s}" + "".join(cols))


def print_missing(
    coverage: dict[int, dict[str, int]],
    articulations: list[str],
) -> None:
    """Print notes missing for each articulation and flag large gaps."""
    print("\nMissing notes by articulation")
    print("(gap > 2 semitones marked with ⚠  — re-recording recommended)\n")

    for art in articulations:
        present = sorted(
            m for m in range(CELLO_MIDI_MIN, CELLO_MIDI_MAX + 1)
            if coverage.get(m, {}).get(art, 0) > 0
        )
        missing = [
            m for m in range(CELLO_MIDI_MIN, CELLO_MIDI_MAX + 1)
            if coverage.get(m, {}).get(art, 0) == 0
        ]

        if not missing:
            print(f"  {art:12s}  ✓ complete coverage")
            continue

        # Find gaps between consecutive present notes.
        gaps: dict[int, int] = {}   # missing_midi → nearest present distance
        for m in missing:
            if not present:
                gaps[m] = 999
                continue
            nearest = min(present, key=lambda p: abs(p - m))
            gaps[m] = abs(nearest - m)

        large_gap_notes  = [m for m in missing if gaps[m] > ACCEPTABLE_GAP]
        small_gap_notes  = [m for m in missing if gaps[m] <= ACCEPTABLE_GAP]

        parts = []
        if large_gap_notes:
            parts.append("⚠  " + ", ".join(midi_to_name(m) for m in large_gap_notes))
        if small_gap_notes:
            parts.append("   " + ", ".join(midi_to_name(m) for m in small_gap_notes))

        print(f"  {art:12s}  missing: {len(missing)} notes")
        for p in parts:
            print(f"              {p}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_coverage",
        description="Report sample coverage across the cello range.",
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
        help="Articulations to check (default: all four).",
    )
    p.add_argument(
        "--takes",
        action="store_true",
        default=False,
        help="Show take counts instead of presence symbols in the table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rows     = load_index(args.output_dir)
    coverage = analyse(rows, args.articulations)

    n_present = sum(
        1 for m in range(CELLO_MIDI_MIN, CELLO_MIDI_MAX + 1)
        if any(coverage.get(m, {}).get(a, 0) > 0 for a in args.articulations)
    )
    n_total = CELLO_MIDI_MAX - CELLO_MIDI_MIN + 1

    print(f"\nOutput directory : {args.output_dir}")
    print(f"Articulations    : {', '.join(args.articulations)}")
    print(f"Notes with samples: {n_present} / {n_total} "
          f"({100 * n_present / n_total:.0f}% of cello range)")

    print_chromatic_map(coverage, args.articulations)
    print_articulation_table(coverage, args.articulations, args.takes)
    print_missing(coverage, args.articulations)

    return 0


if __name__ == "__main__":
    sys.exit(main())
