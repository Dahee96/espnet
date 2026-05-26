#!/usr/bin/env bash
# egs2/libritts_r/enh1/run.sh
# Speech Cleaner — Sidon-based speech restoration recipe
#
# Stages:
#  1  Data prep       : local/data.sh → FP/VOC wav.scp + noise pool
#  2  Noisy pairs 16k : FP training data  (all datasets → 16kHz)
#  3  Noisy pairs 48k : VOC training data (48kHz datasets only, no upsample)
#  4  collect_stats   : Feature Predictor
#  5  FP training     : Stage 1 (standard Trainer)
#  6  collect_stats   : Vocoder pretrain
#  7  VOC pretrain    : Stage 2 (GT SSL features, GANTrainer)
#  8  collect_stats   : Vocoder finetune
#  9  VOC finetune    : Stage 3 (predicted SSL features, GANTrainer)
# 10  Inference       : LibriTTS test-clean / test-other
# 11  Scoring         : WER / DNSMOS / NISQA / SpkSim

set -euo pipefail
. ./path.sh || exit 1
. ./cmd.sh  || exit 1
. ./db.sh   || exit 1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Defaults ──────────────────────────────────────────────────────────────
stage=1
stop_stage=11
ngpu=4
nj=64
python=python3

fp_config=conf/train_speech_cleaner_fp_xeus.yaml #conf/train_speech_cleaner_fp.yaml
ssl_encoder=$(grep "^ssl_encoder:" ${fp_config} | awk '{print $2}')
fp_exp=exp/speech_cleaner_fp_${ssl_encoder}_8s

voc_pretrain_config=conf/train_speech_cleaner_voc_pretrain.yaml
voc_pretrain_exp=exp/speech_cleaner_voc_pretrain

voc_finetune_config=conf/train_speech_cleaner_voc_finetune.yaml
voc_finetune_exp=exp/speech_cleaner_voc_finetune

restored_dir=exp/restored
test_sets="test-clean test-other"

. utils/parse_options.sh || exit 1

# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Data preparation
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation"
    bash local/data.sh
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 2: Noisy pairs at 16 kHz  (Feature Predictor)
# ALL datasets, downsampled to 16 kHz by prepare_noisy_data.py
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Noisy data generation (16 kHz, FP)"
    for split in train dev; do
        src=data/${split}_fp
        reps=4; [ "${split}" = "dev" ] && reps=1
        ${python} local/prepare_noisy_data.py \
            --clean_wav_scp "${src}/wav.scp" \
            --noise_dir     data/noise_pool \
            --out_dir       data/${split}_paired_16k \
            --n_repeat      ${reps} \
            --sr            16000 \
            --nj            ${nj}
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 3: Noisy pairs at 48 kHz  (Vocoder)
# 48kHz datasets ONLY — no upsampling of 24kHz data
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Noisy data generation (48 kHz, VOC)"
    if [ ! -s data/train_voc/wav.scp ]; then
        log "WARNING: data/train_voc/wav.scp is empty. Skipping Stage 3."
    else
        for split in train dev; do
            src=data/${split}_voc
            reps=4; [ "${split}" = "dev" ] && reps=1
            ${python} local/prepare_noisy_data.py \
                --clean_wav_scp "${src}/wav.scp" \
                --noise_dir     data/noise_pool \
                --out_dir       data/${split}_paired_48k \
                --n_repeat      ${reps} \
                --sr            48000 \
                --nj            ${nj}
        done
    fi
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 4: collect_stats — Feature Predictor
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: collect_stats (feature predictor)"
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_fp \
        --config "${fp_config}" \
        --train_data_path_and_name_and_type "data/train_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_16k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_16k/clean/wav.scp,speech_ref1,sound" \
        --output_dir "${fp_exp}" \
        --collect_stats true \
        --ngpu 0
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 5: Feature Predictor training
# ─────────────────────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Feature Predictor training"
    ${cuda_cmd} --gpu ${ngpu} "${fp_exp}/train.log" \
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_fp \
        --config "${fp_config}" \
        --train_data_path_and_name_and_type "data/train_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_16k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_16k/clean/wav.scp,speech_ref1,sound" \
        --train_shape_file "${fp_exp}/train/noisy_speech_shape" \
        --valid_shape_file "${fp_exp}/valid/noisy_speech_shape" \
        --output_dir "${fp_exp}" \
        --ngpu ${ngpu} \
        --multiprocessing_distributed true \
        --unused_parameters true \
        --resume true
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 6: collect_stats — Vocoder pretrain
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: collect_stats (vocoder pretrain)"
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "${voc_pretrain_config}" \
        --train_data_path_and_name_and_type "data/train_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --output_dir "${voc_pretrain_exp}" \
        --collect_stats true \
        --ngpu 0
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 7: Vocoder pretrain  (GT SSL features)
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Vocoder pretrain (GT SSL features)"
    ${cuda_cmd} --gpu ${ngpu} "${voc_pretrain_exp}/train.log" \
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "${voc_pretrain_config}" \
        --train_data_path_and_name_and_type "data/train_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --train_shape_file "${voc_pretrain_exp}/train/noisy_speech_shape" \
        --valid_shape_file "${voc_pretrain_exp}/valid/noisy_speech_shape" \
        --output_dir "${voc_pretrain_exp}" \
        --ngpu ${ngpu} \
        --multiprocessing_distributed true \
        --unused_parameters true \
        --generator_first false \
        --resume true
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 8: collect_stats — Vocoder finetune
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: collect_stats (vocoder finetune)"
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "${voc_finetune_config}" \
        --fp_model_path "${fp_exp}/valid.loss.best.pth" \
        --train_data_path_and_name_and_type "data/train_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --output_dir "${voc_finetune_exp}" \
        --collect_stats true \
        --ngpu 0
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 9: Vocoder finetune  (predicted SSL features)
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "Stage 9: Vocoder finetune (predicted SSL features)"
    ${cuda_cmd} --gpu ${ngpu} "${voc_finetune_exp}/train.log" \
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "${voc_finetune_config}" \
        --fp_model_path "${fp_exp}/valid.loss.best.pth" \
        --init_param "${voc_pretrain_exp}/valid.loss.G.best.pth:vocoder:vocoder" \
        --train_data_path_and_name_and_type "data/train_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --train_data_path_and_name_and_type "data/train_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/noisy/wav.scp,noisy_speech,sound" \
        --valid_data_path_and_name_and_type "data/dev_paired_48k/clean/wav.scp,speech_ref1,sound" \
        --train_shape_file "${voc_finetune_exp}/train/noisy_speech_shape" \
        --valid_shape_file "${voc_finetune_exp}/valid/noisy_speech_shape" \
        --output_dir "${voc_finetune_exp}" \
        --ngpu ${ngpu} \
        --multiprocessing_distributed true \
        --unused_parameters true \
        --generator_first false \
        --resume true
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 10: Inference
# Input : LibriTTS ORIGINAL test sets (24kHz, not restored)
# Output: 48kHz restored speech
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    log "Stage 10: Inference"
    for tset in ${test_sets}; do
        log "  ${tset}"
        # Downsample LibriTTS 24kHz → 16kHz for SSL encoder input
        ${python} local/upsample_wav_scp.py \
            --wav_scp   data/libritts_${tset}/wav.scp \
            --out_dir   data/libritts_${tset}_16k \
            --target_sr 16000 \
            --nj        ${nj}

        ${python} -m espnet2.bin.enh_inference_speech_cleaner \
            --fp_train_config  "${fp_exp}/config.yaml" \
            --fp_model_file    "${fp_exp}/valid.loss.best.pth" \
            --voc_train_config "${voc_finetune_exp}/config.yaml" \
            --voc_model_file   "${voc_finetune_exp}/valid.loss.G.best.pth" \
            --wav_scp          "data/libritts_${tset}_16k/wav.scp" \
            --output_dir       "${restored_dir}/${tset}" \
            --batch_size 8 --device cuda --dtype bfloat16
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 11: Scoring
# Compare restored output vs LibriTTS-R (Miipher) reference
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Scoring"
    for tset in ${test_sets}; do
        ${python} local/score.py \
            --restored_dir  "${restored_dir}/${tset}" \
            --ref_wav_scp   "data/libritts_r_${tset}/wav.scp" \
            --noisy_wav_scp "data/libritts_${tset}/wav.scp" \
            --output_dir    "exp/scores/${tset}" \
            --nj            ${nj}
        log "  Scores: exp/scores/${tset}/scores.json"
    done
fi

log "=== Done (stage ${stage} ~ ${stop_stage}) ==="