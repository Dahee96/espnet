#!/usr/bin/env bash
# local/data.sh
# Data preparation for Speech Cleaner (Sidon-based) recipe.
#
# Reads DATASET_* and NOISE_* variables already exported by db.sh via run.sh.
# SR is hardcoded here per dataset (not in db.sh) following ESPnet convention.
# To add a new dataset: (1) add path in db.sh, (2) add its SR in _dataset_sr().
#
# Outputs:
#   data/train_fp/wav.scp     ALL datasets → Feature Predictor (will be → 16k)
#   data/dev_fp/wav.scp       LibriTTS-R dev → FP validation
#   data/train_voc/wav.scp    48kHz datasets ONLY → Vocoder (native, no upsample)
#   data/dev_voc/wav.scp      48kHz dev proxy
#   data/noise_pool/          merged symlinks from all NOISE_* dirs
#   data/libritts_{test-clean,test-other}/wav.scp   inference input

set -euo pipefail
log() { echo "[data.sh $(date '+%H:%M:%S')] $*"; }

# ── Known sample rates per DATASET_* variable name ────────────────────────
# Add new datasets here when extending the recipe.
_dataset_sr() {
    case "$1" in
        DATASET_LIBRITTS_R)   echo 24000 ;;
        DATASET_JVS)          echo 24000 ;;
        DATASET_FLEURS_R)     echo 24000 ;;
        DATASET_VCTK)         echo 48000 ;;
        DATASET_EARS)         echo 48000 ;;
        DATASET_EXPRESSO)     echo 48000 ;;
        DATASET_HIFICAPTAIN)  echo 48000 ;;
        DATASET_JSUT)         echo 48000 ;;
        DATASET_BIBLETTS)     echo 48000 ;;
        *)                    echo 0     ;;
    esac
}

# ── Find audio files under a directory ────────────────────────────────────
_find_audio() {
    find "$1" \( -name "*.wav" -o -name "*.flac" \) \
        | grep -v "_original\.wav" | sort
}

# ── Build Kaldi-format dir from a file listing one audio path per line ────
_make_kaldi() {
    local flist=$1 out=$2
    mkdir -p "${out}"
    awk '{n=split($0,a,"/"); f=a[n]; spk=a[n-1];
          sub(/\.(wav|flac)$/,"",f); gsub(/_restored$/,"",f);
          uttid=spk"_"f; gsub(/[^A-Za-z0-9_-]/,"_",uttid);
          print uttid, $0}' "${flist}" | sort -u > "${out}/wav.scp"
    awk '{u=$1; n=split(u,p,"_"); print $1, p[1]}' \
        "${out}/wav.scp" | sort -u > "${out}/utt2spk"
    python3 - "${out}" <<'PY'
import sys; from collections import defaultdict
d=defaultdict(list)
[d[s].append(u) for l in open(sys.argv[1]+"/utt2spk") for u,s in [l.strip().split()]]
open(sys.argv[1]+"/spk2utt","w").writelines(s+" "+" ".join(sorted(d[s]))+"\n" for s in sorted(d))
PY
    log "  $(wc -l < ${out}/wav.scp) utts → ${out}"
}

# ─────────────────────────────────────────────────────────────────────────
# 1. LibriTTS-R  (mandatory — provides the dev split)
# ─────────────────────────────────────────────────────────────────────────
[ -n "${DATASET_LIBRITTS_R:-}" ] && [ -d "${DATASET_LIBRITTS_R}" ] \
    || { log "ERROR: DATASET_LIBRITTS_R not set or missing."; exit 1; }

log "LibriTTS-R: ${DATASET_LIBRITTS_R}"

tmp_ltr_train=$(mktemp); tmp_ltr_dev=$(mktemp)
for sub in train-clean-100 train-clean-360 train-other-500; do
    src="${DATASET_LIBRITTS_R}/${sub}"
    [ -d "${src}" ] && _find_audio "${src}" >> "${tmp_ltr_train}" \
                    || log "  SKIP ${sub}"
done
for sub in dev-clean dev-other; do
    src="${DATASET_LIBRITTS_R}/${sub}"
    [ -d "${src}" ] && _find_audio "${src}" >> "${tmp_ltr_dev}" \
                    || log "  SKIP ${sub}"
done

# ─────────────────────────────────────────────────────────────────────────
# 2. Other datasets
# ─────────────────────────────────────────────────────────────────────────
tmp_fp_extra=$(mktemp)    # non-LibriTTS-R audio for FP train
tmp_voc_train=$(mktemp)   # 48kHz audio for VOC train
tmp_voc_dev=$(mktemp)     # 48kHz audio for VOC dev (proxy)

for ds_var in $(compgen -v | grep '^DATASET_' | sort); do
    [ "${ds_var}" = "DATASET_LIBRITTS_R" ] && continue
    ds_dir="${!ds_var:-}"
    [ -z "${ds_dir}" ] && continue
    [ -d "${ds_dir}" ] || { log "SKIP ${ds_var}: not found"; continue; }

    ds_sr=$(_dataset_sr "${ds_var}")
    log "${ds_var} (${ds_sr} Hz): ${ds_dir}"

    tmp_ds=$(mktemp)
    _find_audio "${ds_dir}" > "${tmp_ds}"

    # All datasets → FP train
    cat "${tmp_ds}" >> "${tmp_fp_extra}"

    # 48kHz only → VOC
    if [ "${ds_sr}" = "48000" ]; then
        head -200 "${tmp_ds}" >> "${tmp_voc_dev}"   # proxy dev
        cat "${tmp_ds}"        >> "${tmp_voc_train}"
    fi
    rm -f "${tmp_ds}"
done

# ─────────────────────────────────────────────────────────────────────────
# 3. Build output Kaldi dirs
# ─────────────────────────────────────────────────────────────────────────
log "Building data/train_fp ..."
cat "${tmp_ltr_train}" "${tmp_fp_extra}" | sort -u > /tmp/_sc_fp_train
_make_kaldi /tmp/_sc_fp_train data/train_fp

log "Building data/dev_fp ..."
_make_kaldi "${tmp_ltr_dev}" data/dev_fp

log "Building data/train_voc ..."
if [ -s "${tmp_voc_train}" ]; then
    _make_kaldi "${tmp_voc_train}" data/train_voc
else
    log "WARNING: No 48kHz datasets found — data/train_voc is empty."
    mkdir -p data/train_voc; > data/train_voc/wav.scp
fi

log "Building data/dev_voc ..."
if [ -s "${tmp_voc_dev}" ]; then
    _make_kaldi "${tmp_voc_dev}" data/dev_voc
else
    log "WARNING: No 48kHz dev data — using first 200 utts from train_voc."
    mkdir -p data/dev_voc
    head -200 data/train_voc/wav.scp | awk '{print $2}' > "${tmp_voc_dev}" || true
    _make_kaldi "${tmp_voc_dev}" data/dev_voc
fi

rm -f "${tmp_ltr_train}" "${tmp_ltr_dev}" "${tmp_fp_extra}" \
      "${tmp_voc_train}" "${tmp_voc_dev}" /tmp/_sc_fp_train

# ─────────────────────────────────────────────────────────────────────────
# 4. Noise pool  (merge all NOISE_* dirs via symlinks)
# ─────────────────────────────────────────────────────────────────────────
log "Building data/noise_pool ..."
mkdir -p data/noise_pool
noise_found=0
for noise_var in $(compgen -v | grep '^NOISE_' | sort); do
    noise_dir="${!noise_var:-}"
    [ -z "${noise_dir}" ] && continue
    [ -d "${noise_dir}" ] || { log "  SKIP ${noise_var}: not found"; continue; }
    log "  ${noise_var}: ${noise_dir}"
    find "${noise_dir}" \( -name "*.wav" -o -name "*.flac" \) \
        | while read -r f; do
            ln -sf "${f}" "data/noise_pool/${noise_var}_$(basename ${f})" 2>/dev/null || true
        done
    noise_found=1
done
[ "${noise_found}" -eq 0 ] && log "WARNING: No noise sources found."
log "  noise pool: $(ls data/noise_pool | wc -l) files"

# ─────────────────────────────────────────────────────────────────────────
# 5. LibriTTS ORIGINAL test sets  (inference input — not restored)
# ─────────────────────────────────────────────────────────────────────────
if [ -n "${LIBRITTS:-}" ] && [ -d "${LIBRITTS}" ]; then
    for s in test-clean test-other; do
        src="${LIBRITTS}/${s}"
        [ -d "${src}" ] || { log "SKIP LibriTTS ${s}"; continue; }
        tmp_t=$(mktemp)
        _find_audio "${src}" > "${tmp_t}"
        _make_kaldi "${tmp_t}" "data/libritts_${s}"
        rm -f "${tmp_t}"
    done
else
    log "SKIP: LIBRITTS not set or missing (needed only for inference)"
fi

log "Done."
log "  FP  train : $(wc -l < data/train_fp/wav.scp) utts"
log "  FP  dev   : $(wc -l < data/dev_fp/wav.scp) utts"
log "  VOC train : $(wc -l < data/train_voc/wav.scp) utts"
log "  VOC dev   : $(wc -l < data/dev_voc/wav.scp) utts"
log "  noise pool: $(ls data/noise_pool | wc -l) files"