#!/usr/bin/env python3
"""Build segments + text for per-headset AMI IHM longform data.

Use this when data/ami_ihm_longform/wav.scp has ONE RECORDING PER HEADSET
(not a mixed/averaged signal), e.g.:

    AMI_ES2004a_H00 sox /DB/AMI/ES2004a/audio/ES2004a.Headset-0.wav ... |
    AMI_ES2004a_H01 sox /DB/AMI/ES2004a/audio/ES2004a.Headset-1.wav ... |
    AMI_ES2004a_H02 sox /DB/AMI/ES2004a/audio/ES2004a.Headset-2.wav ... |
    AMI_ES2004a_H03 sox /DB/AMI/ES2004a/audio/ES2004a.Headset-3.wav ... |

This reuses the exact same parse_words_xml() + group_into_segments() logic
from local/prepare_ami_longform.py, but instead of mixing all 4 headsets
into one signal, it builds segments per-headset and matches each AMI words
XML channel (A/B/C/D) to its corresponding recording_id (H00/H01/H02/H03).

This gives oracle (forced-alignment-derived) speech segment boundaries per
speaker, matching Samuele's "oracle VAD for each speaker" recommendation —
no VAD model needed since AMI provides ground-truth word-level timing.

Recording-id naming is inferred from existing wav.scp entries to guarantee
an exact match — you do not need to hand-edit the AMI_<meeting>_H0<idx>
pattern if your wav.scp uses something slightly different (e.g. a different
prefix); see --recording_id_template.

Usage:
    python local/make_ami_ihm_segments.py \
        --wav_scp     data/ami_ihm_longform/wav.scp \
        --words_dir   /DB/AMI/words \
        --out_dir     data/ami_ihm_longform \
        --max_gap     0.5 \
        --max_dur     20.0 \
        --min_dur     0.5

Outputs (written into --out_dir, alongside the existing wav.scp):
    segments   — utt_id rec_id start end
    text       — utt_id transcript   (per-segment, NOT full-meeting concat)
"""

import argparse
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CHAN2HEADSET = {"A": 0, "B": 1, "C": 2, "D": 3}


# =============================================================================
# I/O
# =============================================================================

def read_wav_scp(path: str) -> Dict[str, str]:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def parse_words_xml(xml_path: str) -> List[Tuple[float, float, str]]:
    """Parse AMI words XML → list of (starttime, endtime, word). Identical
    to the version in local/prepare_ami_longform.py."""
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
    """Identical to local/prepare_ami_longform.py's version: merges adjacent
    words into segments, breaking on silence gaps > max_gap or once a
    segment would exceed max_dur."""
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


# =============================================================================
# Recording-id matching
# =============================================================================

_REC_ID_RE = re.compile(r"^(.+?)_H0(\d)$")


def infer_meeting_and_headset(rec_id: str) -> Optional[Tuple[str, int]]:
    """Parse 'AMI_ES2004a_H00' -> ('ES2004a', 0).

    Strips a leading 'AMI_' prefix if present, then expects '<meeting>_H0<idx>'.
    Adjust this function if your wav.scp uses a different naming convention.
    """
    base = rec_id
    if base.startswith("AMI_"):
        base = base[len("AMI_"):]
    m = _REC_ID_RE.match(base)
    if not m:
        return None
    meeting, hidx = m.groups()
    return meeting, int(hidx)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_scp",   required=True,
                   help="Existing per-headset wav.scp "
                        "(e.g. data/ami_ihm_longform/wav.scp)")
    p.add_argument("--words_dir", required=True,
                   help="AMI words/ directory containing "
                        "<meeting>.<chan>.words.xml files")
    p.add_argument("--out_dir",   required=True,
                   help="Where to write segments + text "
                        "(usually the same dir as wav_scp)")
    p.add_argument("--max_gap",   type=float, default=0.5)
    p.add_argument("--max_dur",   type=float, default=20.0)
    p.add_argument("--min_dur",   type=float, default=0.5)
    args = p.parse_args()

    wav_map = read_wav_scp(args.wav_scp)
    logger.info("%d recordings in wav.scp", len(wav_map))

    # Build {(meeting, headset_idx): rec_id} lookup from the actual wav.scp,
    # so naming mismatches are caught explicitly rather than silently
    # producing zero segments.
    rec_lookup: Dict[Tuple[str, int], str] = {}
    unmatched_recs = []
    for rec_id in wav_map:
        parsed = infer_meeting_and_headset(rec_id)
        if parsed is None:
            unmatched_recs.append(rec_id)
            continue
        rec_lookup[parsed] = rec_id

    if unmatched_recs:
        logger.warning(
            "%d recording ids did not match the expected '<meeting>_H0<idx>' "
            "pattern and will have no segments: %s%s",
            len(unmatched_recs), unmatched_recs[:5],
            " ..." if len(unmatched_recs) > 5 else "",
        )
    logger.info("Matched %d (meeting, headset) pairs from wav.scp", len(rec_lookup))

    meetings = sorted({m for m, _ in rec_lookup})
    logger.info("Meetings found: %s", meetings)

    seg_lines  = []
    text_lines = []
    n_no_xml   = 0

    for meeting in meetings:
        for chan, hidx in CHAN2HEADSET.items():
            rec_id = rec_lookup.get((meeting, hidx))
            if rec_id is None:
                # This headset isn't in wav.scp for this meeting — skip
                continue

            xml_path = os.path.join(args.words_dir, f"{meeting}.{chan}.words.xml")
            if not os.path.exists(xml_path):
                logger.debug("No words XML for %s chan %s (%s)", meeting, chan, rec_id)
                n_no_xml += 1
                continue

            words = parse_words_xml(xml_path)
            segs  = group_into_segments(
                words, max_gap=args.max_gap, max_dur=args.max_dur, min_dur=args.min_dur,
            )

            for start, end, transcript in segs:
                utt_id = (f"{rec_id}_"
                         f"{int(round(start*100)):07d}_"
                         f"{int(round(end*100)):07d}")
                seg_lines.append((utt_id, rec_id, start, end))
                text_lines.append((utt_id, transcript.upper()))

        if meeting == meetings[0] or meetings.index(meeting) % 5 == 0:
            logger.info("  processed %s ...", meeting)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path  = out_dir / "segments"
    text_path = out_dir / "text"

    with open(seg_path, "w") as f:
        for utt_id, rec_id, start, end in sorted(seg_lines):
            f.write(f"{utt_id} {rec_id} {start:.3f} {end:.3f}\n")

    with open(text_path, "w", encoding="utf-8") as f:
        for utt_id, text in sorted(text_lines):
            f.write(f"{utt_id} {text}\n")

    logger.info("Wrote %d segments to %s", len(seg_lines), seg_path)
    logger.info("Wrote %d text entries to %s", len(text_lines), text_path)
    if n_no_xml:
        logger.warning("%d (meeting, headset) pairs had no matching words XML", n_no_xml)
    logger.info("Done. These segments are per-speaker (per-headset), oracle "
                "(forced-alignment derived) — matches Samuele's recommendation "
                "without needing a separate VAD pass.")


if __name__ == "__main__":
    main()