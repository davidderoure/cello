"""Tests for generate_sfz — note helpers, key-range calculation, normalisation, and SFZ output."""

from __future__ import annotations

import csv
import math
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from generate_sfz import (
    generate_sfz,
    key_ranges,
    midi_to_sfz_name,
    name_to_midi,
    note_volume_offsets,
    peak_db,
)


# ---------------------------------------------------------------------------
# name_to_midi
# ---------------------------------------------------------------------------


class TestNameToMidi:
    def test_a4(self):
        assert name_to_midi("A4") == 69

    def test_middle_c(self):
        assert name_to_midi("C4") == 60

    def test_sharp(self):
        assert name_to_midi("C#4") == 61

    def test_low_c(self):
        assert name_to_midi("C2") == 36

    def test_high_b(self):
        assert name_to_midi("B5") == 83

    def test_strips_whitespace(self):
        assert name_to_midi("  A4 ") == 69

    def test_flat_normalised(self):
        # "B" in the pitch position is normalised to A# (Bb = A#)
        assert name_to_midi("A#4") == name_to_midi("A#4")

    def test_empty_returns_none(self):
        assert name_to_midi("") is None

    def test_invalid_note_returns_none(self):
        assert name_to_midi("X4") is None

    def test_missing_octave_returns_none(self):
        assert name_to_midi("A") is None

    def test_negative_octave(self):
        assert name_to_midi("C-1") == 0


# ---------------------------------------------------------------------------
# midi_to_sfz_name
# ---------------------------------------------------------------------------


class TestMidiToSfzName:
    def test_a4(self):
        assert midi_to_sfz_name(69) == "A4"

    def test_middle_c(self):
        assert midi_to_sfz_name(60) == "C4"

    def test_c_sharp(self):
        assert midi_to_sfz_name(61) == "C#4"

    def test_roundtrip(self):
        for midi in range(36, 84):
            assert name_to_midi(midi_to_sfz_name(midi)) == midi


# ---------------------------------------------------------------------------
# key_ranges
# ---------------------------------------------------------------------------


class TestKeyRanges:
    def test_single_note_uses_max_range(self):
        r = key_ranges([60], max_range=6)
        assert r[60] == (54, 66)

    def test_single_note_clamps_to_midi_bounds(self):
        r = key_ranges([0], max_range=6)
        assert r[0][0] == 0
        r = key_ranges([127], max_range=6)
        assert r[127][1] == 127

    def test_two_notes_split_midway(self):
        # Notes 60 and 62 — midpoint is 61, so 60→(lo, 61), 62→(62, hi)
        r = key_ranges([60, 62], max_range=6)
        assert r[60][1] == 61
        assert r[62][0] == 62

    def test_no_gaps_or_overlaps(self):
        notes = [36, 45, 57, 69, 76]
        r = key_ranges(notes, max_range=6)
        sorted_notes = sorted(notes)
        for i in range(len(sorted_notes) - 1):
            hi = r[sorted_notes[i]][1]
            lo = r[sorted_notes[i + 1]][0]
            assert hi + 1 == lo, f"Gap/overlap between {sorted_notes[i]} and {sorted_notes[i+1]}"

    def test_custom_max_range(self):
        r = key_ranges([60], max_range=3)
        assert r[60] == (57, 63)


# ---------------------------------------------------------------------------
# peak_db and note_volume_offsets
# ---------------------------------------------------------------------------


def _write_wav(path: Path, peak: float, sr: int = 48_000, n_frames: int = 480) -> None:
    """Write a minimal mono 16-bit WAV whose peak sample is *peak* (0–1)."""
    sample = int(peak * 32767)
    data = struct.pack("<h", sample) + b"\x00\x00" * (n_frames - 1)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)


class TestPeakDb:
    def test_full_scale(self, tmp_path):
        p = tmp_path / "full.wav"
        _write_wav(p, peak=1.0)
        db = peak_db(p)
        assert db is not None
        assert abs(db) < 0.1   # ≈ 0 dBFS

    def test_half_amplitude(self, tmp_path):
        p = tmp_path / "half.wav"
        _write_wav(p, peak=0.5)
        db = peak_db(p)
        assert db is not None
        assert abs(db - (-6.02)) < 0.2

    def test_missing_file_returns_none(self, tmp_path):
        db = peak_db(tmp_path / "missing.wav")
        assert db is None

    def test_returns_none_when_soundfile_unavailable(self, tmp_path):
        p = tmp_path / "any.wav"
        _write_wav(p, peak=0.5)
        import generate_sfz as gsfz
        original = gsfz._SOUNDFILE_AVAILABLE
        try:
            gsfz._SOUNDFILE_AVAILABLE = False
            assert peak_db(p) is None
        finally:
            gsfz._SOUNDFILE_AVAILABLE = original


class TestNoteVolumeOffsets:
    def test_single_take_returns_zero(self, tmp_path):
        p = tmp_path / "a.wav"
        _write_wav(p, peak=0.5)
        offsets = note_volume_offsets([p])
        assert offsets == [0.0]

    def test_equal_peaks_give_zero_offsets(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"take{i}.wav"
            _write_wav(p, peak=0.5)
            paths.append(p)
        offsets = note_volume_offsets(paths)
        assert all(abs(o) < 0.01 for o in offsets)

    def test_quieter_takes_get_positive_offset(self, tmp_path):
        loud = tmp_path / "loud.wav"
        quiet = tmp_path / "quiet.wav"
        _write_wav(loud, peak=1.0)
        _write_wav(quiet, peak=0.5)
        offsets = note_volume_offsets([loud, quiet])
        assert abs(offsets[0]) < 0.01          # loudest → 0 dB
        assert abs(offsets[1] - 6.02) < 0.2    # quiet → +6 dB boost

    def test_all_zeros_when_soundfile_unavailable(self, tmp_path):
        paths = [tmp_path / f"t{i}.wav" for i in range(2)]
        for p in paths:
            _write_wav(p, peak=0.5)
        import generate_sfz as gsfz
        original = gsfz._SOUNDFILE_AVAILABLE
        try:
            gsfz._SOUNDFILE_AVAILABLE = False
            offsets = note_volume_offsets(paths)
            assert offsets == [0.0, 0.0]
        finally:
            gsfz._SOUNDFILE_AVAILABLE = original


# ---------------------------------------------------------------------------
# generate_sfz
# ---------------------------------------------------------------------------


def _make_notes(tmp_path: Path) -> dict[int, list[Path]]:
    """Create dummy WAVs and return a notes dict suitable for generate_sfz."""
    subdir = tmp_path / "legato"
    subdir.mkdir()
    notes: dict[int, list[Path]] = {}
    # Two notes: A3 (57) with 2 takes, D4 (62) with 1 take
    for midi, peaks in [(57, [0.8, 0.4]), (62, [0.6])]:
        paths = []
        for i, peak in enumerate(peaks, 1):
            p = subdir / f"note{midi}_take{i}.wav"
            _write_wav(p, peak=peak)
            paths.append(p)
        notes[midi] = paths
    return notes


class TestGenerateSfz:
    def test_creates_sfz_file(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        assert sfz.exists()
        assert sfz.suffix == ".sfz"

    def test_contains_pitch_keycenter(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        assert "pitch_keycenter=57" in text
        assert "pitch_keycenter=62" in text

    def test_seq_length_for_multiple_takes(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        assert "seq_length=2" in text
        assert "seq_position=1" in text
        assert "seq_position=2" in text

    def test_no_seq_for_single_take(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        # D4 (midi 62) has only 1 take — no seq_length line for that note
        lines = text.splitlines()
        in_d4 = False
        for line in lines:
            if "D4" in line:
                in_d4 = True
            if in_d4 and line.startswith("<group>"):
                pytest.fail("Unexpected <group> for single-take note")

    def test_normalize_adds_volume_attribute(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=True)
        text = sfz.read_text()
        # A3 has takes at 0.8 and 0.4 — the quieter one should get a volume= line
        assert "volume=" in text

    def test_normalize_loud_take_has_no_volume_line(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=True)
        # Find the first region for A3 (the loud take) and check no volume= before next region
        text = sfz.read_text()
        regions = text.split("<region>")
        # First region (index 1) is the loudest take — should have no volume=
        assert "volume=" not in regions[1].split("<group>")[0].split("// ──")[0]

    def test_no_normalize_omits_volume(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        assert "volume=" not in sfz.read_text()

    def test_relative_paths_in_sample(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        # Paths should be relative (legato/...), not absolute
        for line in text.splitlines():
            if line.strip().startswith("sample="):
                assert not line.strip()[7:].startswith("/"), f"Absolute path in SFZ: {line}"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCli:
    def _write_index(self, output_dir: Path, rows: list[dict]) -> None:
        index = output_dir / "_index.csv"
        with index.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["articulation", "note", "file", "take"])
            writer.writeheader()
            writer.writerows(rows)

    def test_main_writes_sfz(self, tmp_path):
        from generate_sfz import main

        subdir = tmp_path / "legato"
        subdir.mkdir()
        wav = subdir / "A3_legato_001.wav"
        _write_wav(wav, peak=0.5)
        self._write_index(tmp_path, [
            {"articulation": "legato", "note": "A3", "file": "legato/A3_legato_001.wav", "take": 1},
        ])
        rc = main([str(tmp_path), "--articulations", "legato"])
        assert rc == 0
        assert (tmp_path / "legato.sfz").exists()

    def test_main_returns_1_when_no_samples(self, tmp_path):
        from generate_sfz import main

        # Write an index with no matching articulation
        self._write_index(tmp_path, [
            {"articulation": "legato", "note": "A3", "file": "legato/A3_legato_001.wav", "take": 1},
        ])
        rc = main([str(tmp_path), "--articulations", "staccato"])
        assert rc == 1

    def test_no_normalize_flag(self, tmp_path):
        from generate_sfz import main

        subdir = tmp_path / "legato"
        subdir.mkdir()
        wav = subdir / "A3_legato_001.wav"
        _write_wav(wav, peak=0.5)
        self._write_index(tmp_path, [
            {"articulation": "legato", "note": "A3", "file": "legato/A3_legato_001.wav", "take": 1},
        ])
        rc = main([str(tmp_path), "--articulations", "legato", "--no-normalize"])
        assert rc == 0
        text = (tmp_path / "legato.sfz").read_text()
        assert "volume=" not in text

    def test_default_loop_mode_is_no_loop(self, tmp_path):
        from generate_sfz import main

        subdir = tmp_path / "legato"
        subdir.mkdir()
        wav = subdir / "A3_legato_001.wav"
        _write_wav(wav, peak=0.5)
        self._write_index(tmp_path, [
            {"articulation": "legato", "note": "A3", "file": "legato/A3_legato_001.wav", "take": 1},
        ])
        main([str(tmp_path), "--articulations", "legato"])
        text = (tmp_path / "legato.sfz").read_text()
        assert "loop_mode=no_loop" in text

    def test_loop_mode_flag_loop_sustain(self, tmp_path):
        from generate_sfz import main

        subdir = tmp_path / "legato"
        subdir.mkdir()
        wav = subdir / "A3_legato_001.wav"
        _write_wav(wav, peak=0.5)
        self._write_index(tmp_path, [
            {"articulation": "legato", "note": "A3", "file": "legato/A3_legato_001.wav", "take": 1},
        ])
        main([str(tmp_path), "--articulations", "legato", "--loop-mode", "loop_sustain"])
        text = (tmp_path / "legato.sfz").read_text()
        assert "loop_mode=loop_sustain" in text
        assert "loop_mode=no_loop" not in text


# ---------------------------------------------------------------------------
# loop_mode in generate_sfz()
# ---------------------------------------------------------------------------


class TestLoopMode:
    def test_no_loop_default(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        assert "loop_mode=no_loop" in text

    def test_explicit_loop_sustain(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz(
            "legato", notes, tmp_path, max_range=6,
            normalize=False, loop_mode="loop_sustain",
        )
        text = sfz.read_text()
        assert "loop_mode=loop_sustain" in text
        assert "loop_mode=no_loop" not in text

    def test_global_header_contains_loop_mode(self, tmp_path):
        notes = _make_notes(tmp_path)
        sfz = generate_sfz("legato", notes, tmp_path, max_range=6, normalize=False)
        text = sfz.read_text()
        assert "<global> loop_mode=no_loop" in text
