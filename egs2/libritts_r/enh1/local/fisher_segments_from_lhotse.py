#!/usr/bin/env python3
"""Build per-channel Fisher segments + text DIRECTLY from the Lhotse manifest.

Routes each manifest entry to its PER-CHANNEL recording_id (e.g.
"fe_03_00001-A" / "fe_03_00001-B"), matching wav.scp built by
local/prepare_fisher.sh from the *-A.wav / *-B.wav files — NOT the mixed
mono file. This mirrors AMI IHM's one-recording-per-close-talk-channel
structure, which is required for a fair AMI vs Fisher comparison and for
correct SpkSim (input-channel <-> restored-same-channel).

Channel mapping (channel field in the Lhotse manifest -> wav.scp suffix):
    channel 0 -> "-A"
    channel 1 -> "-B"
This is the standard Fisher/Switchboard convention. --verify_channel_map
checks this assumption against a sample of entries and will warn loudly if
the observed channel values don't look like {0, 1} as expected.

The Lhotse manifest already contains oracle utterance boundaries (each
entry is a human-transcribed utterance with exact start/duration) — this
is a better oracle-VAD signal than running a separate VAD model, and
because it is per-channel by construction, no further channel-routing
ambiguity exists once entries are split by their `channel` field.

Usage:
    # One segment per manifest entry (finest granularity, exact oracle boundaries)
    python local/fisher_segments_from_lhotse.py \
        --lhotse_jsonl /DB/fisher/lhotse_manifests/supervisions_notfixed.jsonl.gz \
        --wav_scp      data/fisher_longform/wav.scp \
        --out_dir      data/fisher_longform \
        --merge_gap    0.0

    # Merge same-speaker entries with gaps <= 0.5s into longer segments
    python local/fisher_segments_from_lhotse.py \
        --lhotse_jsonl /DB/fisher/lhotse_manifests/supervisions_notfixed.jsonl.gz \
        --wav_scp      data/fisher_longform/wav.scp \
        --out_dir      data/fisher_longform \
        --merge_gap    0.5

Outputs:
    <out_dir>/segments  — utt_id rec_id start end      (rec_id = "<base>-A"/"-B")
    <out_dir>/text      — utt_id text
"""

import argparse
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CHANNEL_TO_SUFFIX = {0: "A", 1: "B"}


def _open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def read_wav_scp(path: str):
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def load_manifest(path: str, valid_rec_ids: set, verify_channel_map: bool = True):
    """Returns {per_channel_rec_id: [(start, end, speaker, text), ...]}, sorted.

    per_channel_rec_id is built as f"{recording_id}-{CHANNEL_TO_SUFFIX[channel]}",
    e.g. "fe_03_00001-A". Entries whose resulting rec_id is not in
    valid_rec_ids (i.e. not present in wav.scp) are dropped with a warning
    tally, since that means either the channel mapping is wrong or that
    particular channel file is missing on disk.
    """
    by_rec = defaultdict(list)
    n_lines = n_skipped_empty = n_skipped_not_in_scp = n_skipped_bad_channel = 0
    observed_channels = defaultdict(int)

    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)

            rec_id   = obj.get("recording_id")
            text     = (obj.get("text") or "").strip()
            start    = obj.get("start")
            duration = obj.get("duration")
            channel  = obj.get("channel")
            speaker  = obj.get("speaker", "unk")

            if rec_id is None or start is None or duration is None or not text:
                n_skipped_empty += 1
                continue

            observed_channels[channel] += 1

            suffix = CHANNEL_TO_SUFFIX.get(channel)
            if suffix is None:
                n_skipped_bad_channel += 1
                continue

            per_chan_rec_id = f"{rec_id}-{suffix}"
            if per_chan_rec_id not in valid_rec_ids:
                n_skipped_not_in_scp += 1
                continue

            by_rec[per_chan_rec_id].append(
                (float(start), float(start) + float(duration), str(speaker), text)
            )

    for rec_id in by_rec:
        by_rec[rec_id].sort(key=lambda x: x[0])

    logger.info("Loaded %d manifest lines -> %d per-channel recordings with segments",
                n_lines, len(by_rec))
    logger.info("  skipped: %d empty/missing-field, %d unrecognized channel value, "
                "%d not found in wav.scp",
                n_skipped_empty, n_skipped_bad_channel, n_skipped_not_in_scp)

    if verify_channel_map:
        logger.info("Observed channel values in manifest: %s",
                    dict(observed_channels))
        unexpected = set(observed_channels) - set(CHANNEL_TO_SUFFIX)
        if unexpected:
            logger.warning(
                "Manifest contains channel values %s with NO mapping in "
                "CHANNEL_TO_SUFFIX=%s — those entries were dropped. If Fisher "
                "uses a different channel convention than {0:A, 1:B}, edit "
                "CHANNEL_TO_SUFFIX at the top of this script.",
                unexpected, CHANNEL_TO_SUFFIX,
            )
        if n_skipped_not_in_scp > 0.5 * n_lines:
            logger.warning(
                "More than half of manifest lines did not match any recording "
                "in wav.scp after channel mapping (%d / %d). This usually means "
                "the channel->suffix convention is backwards, or wav.scp was "
                "built from a different file set. Spot-check a few rec_ids "
                "manually before trusting the output.",
                n_skipped_not_in_scp, n_lines,
            )

    return by_rec


def merge_entries(entries, merge_gap: float):
    """Merge consecutive same-speaker entries whose gap <= merge_gap.

    entries: list of (start, end, speaker, text), sorted by start.
    Returns list of (start, end, text).
    """
    if merge_gap <= 0:
        return [(s, e, t) for s, e, _, t in entries]

    merged = []
    cur_start, cur_end, cur_speaker, cur_texts = None, None, None, []

    for start, end, speaker, text in entries:
        if cur_speaker is None:
            cur_start, cur_end, cur_speaker, cur_texts = start, end, speaker, [text]
            continue
        gap = start - cur_end
        if speaker == cur_speaker and gap <= merge_gap:
            cur_end = end
            cur_texts.append(text)
        else:
            merged.append((cur_start, cur_end, " ".join(cur_texts)))
            cur_start, cur_end, cur_speaker, cur_texts = start, end, speaker, [text]

    if cur_speaker is not None:
        merged.append((cur_start, cur_end, " ".join(cur_texts)))

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lhotse_jsonl", required=True)
    p.add_argument("--wav_scp",      required=True,
                   help="Per-channel wav.scp from local/prepare_fisher.sh "
                        "(rec_ids like fe_03_00001-A / fe_03_00001-B)")
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--merge_gap",    type=float, default=0.0,
                   help="Merge consecutive same-speaker entries with gap <= "
                        "this many seconds into one segment. 0 = no merging.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_map = read_wav_scp(args.wav_scp)
    valid_rec_ids = set(wav_map.keys())
    logger.info("%d per-channel recordings in wav.scp", len(valid_rec_ids))

    manifest = load_manifest(args.lhotse_jsonl, valid_rec_ids)

    seg_lines = []
    text_lines = []

    for rec_id, entries in sorted(manifest.items()):
        merged = merge_entries(entries, args.merge_gap)
        for start, end, text in merged:
            utt_id = f"{rec_id}_{int(start*100):06d}-{int(end*100):06d}"
            seg_lines.append((utt_id, rec_id, start, end))
            text_lines.append((utt_id, text))

    seg_path  = out_dir / "segments"
    text_path = out_dir / "text"

    with open(seg_path, "w") as f:
        for utt_id, rec_id, start, end in sorted(seg_lines):
            f.write(f"{utt_id} {rec_id} {start:.3f} {end:.3f}\n")

    with open(text_path, "w", encoding="utf-8") as f:
        for utt_id, text in sorted(text_lines):
            f.write(f"{utt_id} {text}\n")

    n_recs_with_segs = len(manifest)
    n_recs_total     = len(valid_rec_ids)
    logger.info("Wrote %d segments (from %d/%d per-channel recordings, "
                "merge_gap=%.2fs) to %s",
                len(seg_lines), n_recs_with_segs, n_recs_total, args.merge_gap, seg_path)
    logger.info("Wrote %d text entries to %s", len(text_lines), text_path)

    recs_without_segs = valid_rec_ids - set(manifest.keys())
    if recs_without_segs:
        logger.warning("%d per-channel recordings in wav.scp got ZERO segments: %s%s",
                       len(recs_without_segs), sorted(recs_without_segs)[:5],
                       " ..." if len(recs_without_segs) > 5 else "")
        logger.warning("These will be silently absent from scoring unless investigated.")

    logger.info("Done. All segments are oracle (manifest-derived) and per-channel "
                "(close-talk equivalent) — matches AMI IHM's per-headset structure.")


if __name__ == "__main__":
    main()