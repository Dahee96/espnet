#!/usr/bin/env bash
# local/prepare_fisher.sh
#
# Prepares Kaldi-style wav.scp for Fisher using PER-CHANNEL files
# (*-A.wav / *-B.wav), not the mixed mono *.wav.
#
# Each channel is a separate recording_id, matching how AMI IHM uses one
# recording per close-talk headset:
#   fe_03_00001-A    (speaker A's line, this is the "close-talk" channel)
#   fe_03_00001-B    (speaker B's line)
#
# This matches Lhotse manifest's "channel" field (0 -> A, 1 -> B), so that
# fisher_segments_from_lhotse.py can route each transcript entry to the
# correct per-channel recording.
#
# Fisher directory structure assumed:
#   /DB/fisher/fisher_wavs/fe_03_00001-A.wav
#   /DB/fisher/fisher_wavs/fe_03_00001-B.wav
#   ... (mixed fe_03_00001.wav also present but NOT used here)
#
# Usage:
#   bash local/prepare_fisher.sh \
#       --fisher_wav_dir /DB/fisher/fisher_wavs \
#       --out_dir        data/fisher_longform

set -euo pipefail

fisher_wav_dir=/DB/fisher/fisher_wavs
out_dir=data/fisher_longform

. utils/parse_options.sh 2>/dev/null || true

mkdir -p "${out_dir}"

echo "[Fisher] Building wav.scp from per-channel files in ${fisher_wav_dir} ..."

# Only *-A.wav and *-B.wav (skip the mixed mono fe_03_XXXXX.wav)
find "${fisher_wav_dir}" -maxdepth 1 \( -name "*-A.wav" -o -name "*-B.wav" \) \
    | sort \
    | awk '{
        n = split($1, a, "/"); fname = a[n]
        sub(/\.wav$/, "", fname)
        print fname " " $1
    }' > "${out_dir}/wav.scp"

n_wavs=$(wc -l < "${out_dir}/wav.scp")
echo "  Found ${n_wavs} per-channel Fisher files (A + B combined)."

# Sanity: roughly even split between -A and -B
n_a=$(grep -c -- '-A$' "${out_dir}/wav.scp" || true)
n_b=$(grep -c -- '-B$' "${out_dir}/wav.scp" || true)
echo "    -A channels: ${n_a}"
echo "    -B channels: ${n_b}"
if [ "${n_a}" -ne "${n_b}" ]; then
    echo "  WARNING: -A and -B counts differ — check for missing channel files."
fi

# utt2spk, spk2utt (recording_id == speaker_id is fine for long-form)
awk '{print $1, $1}' "${out_dir}/wav.scp" > "${out_dir}/utt2spk"
awk '{print $1, $1}' "${out_dir}/utt2spk" > "${out_dir}/spk2utt"

echo "[Fisher] Done."
echo "  Output: ${out_dir}/wav.scp  (${n_wavs} per-channel recordings)"
echo "Next: python local/fisher_segments_from_lhotse.py --lhotse_jsonl ... --out_dir ${out_dir}"