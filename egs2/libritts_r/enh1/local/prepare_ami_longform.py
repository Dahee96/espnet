#!/usr/bin/env python3
"""Prepare AMI SC test set long-form data.

Creates:
  data/ami_sdm_longform/
    wav.scp     — SDM full meeting (noisy input for inference)
    text        — full meeting transcript (all speakers concatenated)
    utt2spk
    spk2utt

  data/ami_ihm_mix/
    wav.scp     — IHM Headset 0~3 mixed (clean reference, full meeting)
    text        — same as SDM
    utt2spk
    spk2utt
    segments    — segment timing from words XML (for score_ami.py)

  data/ami_ihm_mix/wavs/
    {meeting}.wav   — pre-mixed IHM wav files (Headset 0~3 averaged, 16kHz)

The IHM mix is pre-computed and saved to disk so score_ami.py can
load the full file and slice segments at runtime without recomputing.

SC test meetings: ES2004, ES2014, IS1009, TS3003, TS3007 (a/b/c/d)

Usage:
    python3 local/prepare_ami_longform.py \
        --ami_dir   /DB/AMI \
        --words_dir /DB/AMI/words \
        --out_dir   /path/to/egs2/libritts_r/enh1/data
"""

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

SC_TEST_MEETINGS = [
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "ES2014a", "ES2014b", "ES2014c", "ES2014d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    "TS3007a", "TS3007b", "TS3007c", "TS3007d",
]

CHAN2HEADSET = {"A": 0, "B": 1, "C": 2, "D": 3}
TARGET_SR    = 16000


def find_audio(ami_dir: str, meeting: str, pattern: str) -> Optional[str]:
    audio_dir = Path(ami_dir) / meeting / "audio"
    if not audio_dir.exists():
        audio_dir = Path(ami_dir) / meeting
    for f in audio_dir.glob(pattern):
        return str(f)
    return None


def load_wav_16k(path: str) -> np.ndarray:
    """Load wav → mono float32 at 16kHz."""
    import subprocess
    # Use sox for robust resampling
    cmd = f"sox {path} -r {TARGET_SR} -c 1 -t raw -e float - "
    proc = subprocess.run(cmd, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # fallback: soundfile + librosa
        import librosa
        wav, sr = sf.read(path, always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        return wav.astype(np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def mix_headsets(ami_dir: str, meeting: str) -> Optional[np.ndarray]:
    """Load and average all available headset channels."""
    wavs = []
    for chan, hidx in CHAN2HEADSET.items():
        path = find_audio(ami_dir, meeting, f"*.Headset-{hidx}.wav")
        if path:
            try:
                wav = load_wav_16k(path)
                wavs.append(wav)
            except Exception as e:
                print(f"  WARNING: failed to load Headset-{hidx} for {meeting}: {e}")

    if not wavs:
        return None

    # Pad to same length and average
    max_len = max(len(w) for w in wavs)
    padded  = [np.pad(w, (0, max_len - len(w))) for w in wavs]
    mixed   = np.mean(padded, axis=0).astype(np.float32)

    # Peak normalize to avoid clipping
    peak = np.abs(mixed).max()
    if peak > 1e-6:
        mixed = mixed / peak * 0.9

    return mixed


def parse_words_xml(xml_path: str) -> List[Tuple[float, float, str]]:
    """Parse AMI words XML → list of (starttime, endtime, word)."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        with open(xml_path, "rb") as f:
            content = f.read().decode("iso-8859-1")
        tree = ET.ElementTree(ET.fromstring(content.encode("utf-8")))

    root  = tree.getroot()
    words = []
    for elem in root:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "w":
            continue
        if elem.get("punc", "false") == "true":
            continue
        start = float(elem.get("starttime", -1))
        end   = float(elem.get("endtime",   -1))
        text  = (elem.text or "").strip()
        if text and start >= 0 and end >= 0:
            words.append((start, end, text))

    return sorted(words, key=lambda x: x[0])


def group_into_segments(
    words: List[Tuple[float, float, str]],
    max_gap: float = 0.5,
    max_dur: float = 20.0,
    min_dur: float = 0.5,
) -> List[Tuple[float, float, str]]:
    if not words:
        return []
    segs      = []
    seg_start = words[0][0]
    seg_end   = words[0][1]
    seg_words = [words[0][2]]

    for start, end, word in words[1:]:
        if (start - seg_end) > max_gap or (end - seg_start) > max_dur:
            if seg_end - seg_start >= min_dur and seg_words:
                segs.append((seg_start, seg_end, " ".join(seg_words)))
            seg_start = start
            seg_end   = end
            seg_words = [word]
        else:
            seg_end = max(seg_end, end)
            seg_words.append(word)

    if seg_end - seg_start >= min_dur and seg_words:
        segs.append((seg_start, seg_end, " ".join(seg_words)))
    return segs


def write_data_dir(out_dir: str, wav_scp: dict, text: dict,
                   utt2spk: dict, segments: Optional[dict] = None):
    os.makedirs(out_dir, exist_ok=True)

    spk2utt = defaultdict(list)
    for uttid, spk in utt2spk.items():
        spk2utt[spk].append(uttid)

    with open(os.path.join(out_dir, "wav.scp"), "w") as f:
        for uttid in sorted(wav_scp):
            f.write(f"{uttid} {wav_scp[uttid]}\n")

    with open(os.path.join(out_dir, "text"), "w") as f:
        for uttid in sorted(text):
            f.write(f"{uttid} {text[uttid]}\n")

    with open(os.path.join(out_dir, "utt2spk"), "w") as f:
        for uttid in sorted(utt2spk):
            f.write(f"{uttid} {utt2spk[uttid]}\n")

    with open(os.path.join(out_dir, "spk2utt"), "w") as f:
        for spk in sorted(spk2utt):
            f.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")

    if segments:
        # segments: uttid → (recording_id, start, end, text)
        with open(os.path.join(out_dir, "segments"), "w") as f:
            for uttid in sorted(segments):
                rec_id, start, end = segments[uttid]
                f.write(f"{uttid} {rec_id} {start:.3f} {end:.3f}\n")

    print(f"  Wrote {len(wav_scp)} recordings → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ami_dir",   required=True)
    p.add_argument("--words_dir", default=None)
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--max_gap",   type=float, default=0.5)
    p.add_argument("--max_dur",   type=float, default=20.0)
    p.add_argument("--min_dur",   type=float, default=0.5)
    args = p.parse_args()

    ami_dir   = args.ami_dir
    words_dir = args.words_dir or os.path.join(ami_dir, "words")
    out_dir   = args.out_dir

    # Output dirs
    sdm_dir     = os.path.join(out_dir, "ami_sdm_longform")
    ihm_mix_dir = os.path.join(out_dir, "ami_ihm_mix")
    ihm_wav_dir = os.path.join(ihm_mix_dir, "wavs")
    os.makedirs(ihm_wav_dir, exist_ok=True)

    sdm_wav   = {}
    sdm_text  = {}
    sdm_u2spk = {}

    ihm_wav      = {}
    ihm_text     = {}
    ihm_u2spk    = {}
    ihm_segments = {}  # uttid → (rec_id, start, end)

    for meeting in SC_TEST_MEETINGS:
        print(f"Processing {meeting} ...")

        # ── SDM ──────────────────────────────────────────────────────────
        sdm_path = find_audio(ami_dir, meeting, "*.Array1-01.wav")
        if not sdm_path:
            print(f"  WARNING: SDM not found for {meeting}")
            continue

        sdm_rec_id           = f"AMI_{meeting}_SDM"
        sdm_wav[sdm_rec_id]  = (f"sox {sdm_path} -r {TARGET_SR} -c 1 "
                                 f"-t wavpcm - |")
        sdm_u2spk[sdm_rec_id] = f"AMI_{meeting}"

        # ── IHM mix ──────────────────────────────────────────────────────
        ihm_mix_wav = mix_headsets(ami_dir, meeting)
        if ihm_mix_wav is None:
            print(f"  WARNING: no headset files for {meeting}")
            continue

        ihm_mix_path = os.path.join(ihm_wav_dir, f"{meeting}.wav")
        sf.write(ihm_mix_path, ihm_mix_wav, TARGET_SR)

        ihm_rec_id             = f"AMI_{meeting}_IHM"
        ihm_wav[ihm_rec_id]    = ihm_mix_path
        ihm_u2spk[ihm_rec_id]  = f"AMI_{meeting}"

        # ── Transcript + segments from words XML ─────────────────────────
        all_meeting_words = []
        for chan in ["A", "B", "C", "D"]:
            xml_path = os.path.join(words_dir, f"{meeting}.{chan}.words.xml")
            if not os.path.exists(xml_path):
                continue

            words = parse_words_xml(xml_path)
            all_meeting_words.extend(words)

            # Per-speaker segments for scoring
            segs = group_into_segments(
                words,
                max_gap=args.max_gap,
                max_dur=args.max_dur,
                min_dur=args.min_dur,
            )
            hidx = CHAN2HEADSET[chan]
            for i, (start, end, transcript) in enumerate(segs):
                uttid = (f"AMI_{meeting}_H0{hidx}_{chan}_"
                         f"{int(round(start*100)):07d}_"
                         f"{int(round(end*100)):07d}")
                ihm_segments[uttid] = (ihm_rec_id, start, end)

        # Full meeting transcript (all speakers, sorted by time)
        all_meeting_words.sort(key=lambda x: x[0])
        full_text = " ".join(w[2].upper() for w in all_meeting_words)
        sdm_text[sdm_rec_id]  = full_text
        ihm_text[ihm_rec_id]  = full_text

    # ── Write data dirs ───────────────────────────────────────────────────
    print(f"\nWriting SDM longform → {sdm_dir}")
    write_data_dir(sdm_dir, sdm_wav, sdm_text, sdm_u2spk)

    print(f"Writing IHM mix → {ihm_mix_dir}")
    write_data_dir(ihm_mix_dir, ihm_wav, ihm_text, ihm_u2spk,
                   segments=ihm_segments)

    print(f"\nDone.")
    print(f"  SDM longform : {len(sdm_wav)} recordings")
    print(f"  IHM mix      : {len(ihm_wav)} recordings, "
          f"{len(ihm_segments)} segments")
    print(f"  IHM wavs     : {ihm_wav_dir}")
    print()
    print("Next steps:")
    print(f"  1. Run inference on {sdm_dir}/wav.scp")
    print(f"  2. python3 local/score_ami.py \\")
    print(f"       --restored_scp  exp/restored_xxx_ami_sdm_longform/wav.scp \\")
    print(f"       --noisy_scp     {sdm_dir}/wav.scp \\")
    print(f"       --ihm_mix_scp   {ihm_mix_dir}/wav.scp \\")
    print(f"       --segments      {ihm_mix_dir}/segments \\")
    print(f"       --text          {ihm_mix_dir}/text \\")
    print(f"       --out_dir       exp/scores/xxx_ami_sdm_longform")


if __name__ == "__main__":
    main()