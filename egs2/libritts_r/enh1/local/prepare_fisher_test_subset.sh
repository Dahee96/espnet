#!/usr/bin/env bash
# local/prepare_fisher_test_subset.sh
#
# Builds data/fisher_longform_test_set/ using a FIXED, REPRODUCIBLE subset of
# 61 Fisher conversations (122 per-channel files: 61 x {-A,-B}).
#
# IMPORTANT CAVEAT (read before using for final paper numbers):
#   Samuele's split (from Morrone et al. 2023, arXiv:2303.12002) defines the
#   Fisher test set as 61 conversations chosen so that NO speaker identity
#   overlaps between train/val/test (11577/61/61 split, ~7h test). However,
#   neither that paper nor any later one we could find publishes the exact
#   61 conversation IDs, and there is no official released split for Fisher
#   (confirmed independently by arXiv:2508.07375, which complains about this
#   exact problem and proposes its own ad-hoc split). The speaker-disjoint
#   constraint means the real test set is almost certainly NOT simply
#   "the first 61 conversations" — but since we cannot reconstruct the exact
#   list without Samuele's split file, this script uses the first 61
#   conversations (fe_03_00001 .. fe_03_00061) purely as a NAMED PLACEHOLDER:
#   reproducible, easy to describe, and roughly matching the 59 conversations
#   already present as mixed-mono files in fisher_wavs/.
#
#   REPLACE conversation_ids.txt (see below) with the real list as soon as
#   Samuele shares it, then just re-run this script — nothing else changes.
#
# Usage:
#   bash local/prepare_fisher_test_subset.sh \
#       --fisher_wav_dir /DB/fisher/fisher_wavs \
#       --out_dir        data/fisher_longform_test_set \
#       --n_conversations 61 \
#       --start_idx       1

set -euo pipefail

fisher_wav_dir=/DB/fisher/fisher_wavs
out_dir=data/fisher_longform_test_set
n_conversations=61
start_idx=1
conversation_ids_file=""   # if set, overrides start_idx/n_conversations entirely

. utils/parse_options.sh 2>/dev/null || true

mkdir -p "${out_dir}"

# ── Build the conversation ID list ───────────────────────────────────────────
ids_path="${out_dir}/conversation_ids.txt"

if [ -n "${conversation_ids_file}" ] && [ -f "${conversation_ids_file}" ]; then
    echo "[Fisher test subset] Using provided conversation ID list: ${conversation_ids_file}"
    cp "${conversation_ids_file}" "${ids_path}"
else
    echo "[Fisher test subset] No --conversation_ids_file given."
    echo "  Falling back to PLACEHOLDER subset: first ${n_conversations} conversations"
    echo "  starting at fe_03_$(printf '%05d' ${start_idx})."
    echo "  *** This is NOT Samuele's actual split — replace as soon as available. ***"
    : > "${ids_path}"
    end_idx=$((start_idx + n_conversations - 1))
    for i in $(seq "${start_idx}" "${end_idx}"); do
        printf "fe_03_%05d\n" "${i}" >> "${ids_path}"
    done
fi

n_ids=$(wc -l < "${ids_path}")
echo "  ${n_ids} conversation IDs -> ${ids_path}"

# ── Build wav.scp from the per-channel files for these conversations ───────
wav_scp="${out_dir}/wav.scp"
: > "${wav_scp}"
n_found=0
n_missing=0

while read -r conv_id; do
    for chan in A B; do
        wav_path="${fisher_wav_dir}/${conv_id}-${chan}.wav"
        if [ -f "${wav_path}" ]; then
            echo "${conv_id}-${chan} ${wav_path}" >> "${wav_scp}"
            n_found=$((n_found + 1))
        else
            echo "  WARNING: missing ${wav_path}"
            n_missing=$((n_missing + 1))
        fi
    done
done < "${ids_path}"

sort -o "${wav_scp}" "${wav_scp}"

echo "[Fisher test subset] wav.scp: ${n_found} files found, ${n_missing} missing"
echo "  (expected $(( n_ids * 2 )) = ${n_ids} conversations x 2 channels)"

awk '{print $1, $1}' "${wav_scp}" > "${out_dir}/utt2spk"
awk '{print $1, $1}' "${out_dir}/utt2spk" > "${out_dir}/spk2utt"

echo "[Fisher test subset] Done."
echo "  Output dir: ${out_dir}"
echo "Next: python local/fisher_segments_from_lhotse.py \\"
echo "          --lhotse_jsonl /DB/fisher/lhotse_manifests/supervisions_notfixed.jsonl.gz \\"
echo "          --wav_scp      ${wav_scp} \\"
echo "          --out_dir      ${out_dir} \\"
echo "          --merge_gap    0.5"