#!/usr/bin/env bash
# egs2/libritts_r/enh1/run.sh
# Speech Cleaner — Sidon-based speech restoration recipe
#
# On-the-fly degradation pipeline:
#   Stages 1-2: data prep (clean wav.scps + noise pool + RIR pool)
#   Stage  3:   RIR pool generation  (pre-generate RIRs, ~1h for 50k)
#   Stage  4:   collect_stats FP
#   Stage  5:   FP training
#   Stage  6:   collect_stats VOC pretrain
#   Stage  7:   VOC pretrain  (GT SSL features)
#   Stage  8:   collect_stats VOC finetune
#   Stage  9:   VOC finetune  (predicted SSL features)
#   Stage 10:   Inference
#   Stage 11:   Scoring

set -euo pipefail
. ./path.sh || exit 1
. ./cmd.sh  || exit 1
. ./db.sh   || exit 1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Defaults ──────────────────────────────────────────────────────────────
stage=1
stop_stage=11
ngpu=2 #4
nj=64
python=python3

fp_config=conf/train_speech_cleaner_fp_xeus.yaml
voc_pretrain_config=conf/train_speech_cleaner_voc_pretrain_xeus.yaml
voc_finetune_config=conf/train_speech_cleaner_voc_finetune_xeus.yaml
# fp_config=conf/train_speech_cleaner_fp.yaml
# voc_pretrain_config=conf/train_speech_cleaner_voc_pretrain.yaml
# voc_finetune_config=conf/train_speech_cleaner_voc_finetune.yaml


test_sets="test-clean test-other"

. utils/parse_options.sh || exit 1

# ── Experiment directory naming ───────────────────────────────────────────
# Reads ssl_encoder, use_multilayer_loss, multilayer_mode from yaml configs
# to build human-readable exp directory names.
_exp_suffix() {
    local cfg=$1
    local ssl enc use_ml ml_mode suffix
    ssl=$(grep "^ssl_encoder:" "${cfg}" 2>/dev/null | awk '{print $2}')
    use_ml=$(grep "^use_multilayer_loss:" "${cfg}" 2>/dev/null | awk '{print $2}')
    ml_mode=$(grep "^multilayer_mode:" "${cfg}" 2>/dev/null | awk '{print $2}')
    enc=${ssl:-w2v_bert2}
    if [ "${use_ml:-false}" = "true" ]; then
        suffix="${enc}_multi_${ml_mode:-low}"
    else
        suffix="${enc}_single"
    fi
    echo "${suffix}"
}

_voc_exp_suffix() {
    local cfg=$1
    local ssl enc use_ml ml_mode suffix
    ssl=$(grep "^ssl_encoder:" "${cfg}" 2>/dev/null | awk '{print $2}')
    use_ml=$(grep "^use_multilayer_feat:" "${cfg}" 2>/dev/null | awk '{print $2}')
    ml_mode=$(grep "^multilayer_mode:" "${cfg}" 2>/dev/null | awk '{print $2}')
    enc=${ssl:-w2v_bert2}
    if [ "${use_ml:-false}" = "true" ]; then
        suffix="${enc}_multi_${ml_mode:-low}"
    else
        suffix="${enc}_single"
    fi
    echo "${suffix}"
}

fp_suffix=$(_exp_suffix "${fp_config}")
fp_exp=exp/speech_cleaner_fp_${fp_suffix}_8

voc_suffix=$(_voc_exp_suffix "${voc_pretrain_config}")
voc_pretrain_exp=exp/speech_cleaner_voc_pretrain_${voc_suffix}
voc_finetune_exp=exp/speech_cleaner_voc_finetune_${voc_suffix}

restored_dir=exp/restored_${fp_suffix}

log "FP  exp: ${fp_exp}"
log "VOC pretrain exp: ${voc_pretrain_exp}"
log "VOC finetune exp: ${voc_finetune_exp}"

# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Data preparation (clean wav.scps + noise pool symlinks)
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation"
    bash local/data.sh
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 2: Resample FP wav.scps to 16kHz via sox pipe
# FP training requires all audio at 16kHz.
# Datasets have mixed SRs (LibriTTS-R: 24k, VCTK/EARS: 48k, etc.)
# sox pipe: audio is resampled on-the-fly at load time, no disk copy.
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Resample FP wav.scps to 16kHz (actual wav files)"
    for split in train dev; do
        mkdir -p data/${split}_fp_16k/wav
        ${python} local/resample_wav_scp.py \
            --input_scp  data/${split}_fp/wav.scp \
            --output_scp data/${split}_fp_16k/wav.scp \
            --wav_dir    data/${split}_fp_16k/wav \
            --target_sr  16000 \
            --nj         ${nj}
        log "  ${split}: $(wc -l < data/${split}_fp_16k/wav.scp) utts"
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 3: RIR pool generation
# Pre-generate 50k room impulse responses for on-the-fly reverberation.
# Takes ~1h on 16 CPU workers. Re-run skips already-generated files.
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: RIR pool generation (data/rir_pool)"
    ${python} local/prepare_rir_pool.py \
        --out_dir data/rir_pool \
        --n_rirs  50000 \
        --nj      ${nj}
    log "  RIR pool: $(ls data/rir_pool/*.wav 2>/dev/null | wc -l) files"
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 4: collect_stats — Feature Predictor
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: collect_stats (feature predictor)"
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_fp \
        --config "${fp_config}" \
        --train_data_path_and_name_and_type "data/train_fp_16k/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_fp_16k/wav.scp,speech_ref1,sound" \
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
        --train_data_path_and_name_and_type "data/train_fp_16k/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_fp_16k/wav.scp,speech_ref1,sound" \
        --train_shape_file "${fp_exp}/train/speech_ref1_shape" \
        --valid_shape_file "${fp_exp}/valid/speech_ref1_shape" \
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
        --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
        --output_dir "${voc_pretrain_exp}" \
        --collect_stats true \
        --ngpu 0
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 7: Vocoder pretrain  (GT SSL features)
# ─────────────────────────────────────────────────────────────────────────
# if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
#     log "Stage 7: Vocoder pretrain (GT SSL features)"
#     ${cuda_cmd} --gpu ${ngpu} "${voc_pretrain_exp}/train.log" \
#     ${python} -m espnet2.bin.enh_train_speech_cleaner \
#         --task sc_gan \
#         --config "${voc_pretrain_config}" \
#         --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
#         --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
#         --train_shape_file "${voc_pretrain_exp}/train/speech_ref1_shape" \
#         --valid_shape_file "${voc_pretrain_exp}/valid/speech_ref1_shape" \
#         --output_dir "${voc_pretrain_exp}" \
#         --ngpu ${ngpu} \
#         --multiprocessing_distributed true \
#         --unused_parameters true \
#         --generator_first false \
#         --resume true
# fi
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Vocoder pretrain (GT SSL features)"
    ${cuda_cmd} --gpu ${ngpu} "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/train.log" \
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "conf/train_speech_cleaner_voc_pretrain_xeus_utterance.yaml" \
        --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
        --train_shape_file "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/train/speech_ref1_shape" \
        --valid_shape_file "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/valid/speech_ref1_shape" \
        --output_dir "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic" \
        --ngpu 2 \
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
        --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
        --output_dir "${voc_finetune_exp}" \
        --collect_stats true \
        --ngpu 0
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 9: Vocoder finetune  (predicted SSL features)
# ─────────────────────────────────────────────────────────────────────────
# if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
#     log "Stage 9: Vocoder finetune (predicted SSL features)"
#     ${cuda_cmd} --gpu ${ngpu} "${voc_finetune_exp}/train.log" \
#     ${python} -m espnet2.bin.enh_train_speech_cleaner \
#         --task sc_gan \
#         --config "${voc_finetune_config}" \
#         --fp_model_path "${fp_exp}/valid.loss.best.pth" \
#         --init_param "${voc_pretrain_exp}/valid.loss_G.best.pth:vocoder:vocoder" \
#         --init_param "${voc_pretrain_exp}/valid.loss_G.best.pth:mpd:mpd" \
#         --init_param "${voc_pretrain_exp}/valid.loss_G.best.pth:msd:msd" \
#         --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
#         --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
#         --train_shape_file "${voc_finetune_exp}/train/speech_ref1_shape" \
#         --valid_shape_file "${voc_finetune_exp}/valid/speech_ref1_shape" \
#         --output_dir "${voc_finetune_exp}" \
#         --ngpu ${ngpu} \
#         --multiprocessing_distributed true \
#         --unused_parameters true \
#         --generator_first false \
#         --resume true
# fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "Stage 9: Vocoder finetune (predicted SSL features)"
    ${cuda_cmd} --gpu ${ngpu} "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic_again/train.log" \
    ${python} -m espnet2.bin.enh_train_speech_cleaner \
        --task sc_gan \
        --config "conf/train_speech_cleaner_voc_finetune_xeus_utterance.yaml" \
        --fp_model_path "exp/speech_cleaner_fp_xeus_multi_all_real/valid.loss.best.pth" \
        --init_param "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/valid.stoi.best.pth:vocoder:vocoder" \
        --init_param "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/valid.stoi.best.pth:discriminator:discriminator" \
        --init_param "exp/speech_cleaner_voc_pretrain_xeus_multi_all_utterance_dynamic/valid.stoi.best.pth:layer_router:layer_router" \
        --output_dir "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic_again" \
        --train_data_path_and_name_and_type "data/train_voc/wav.scp,speech_ref1,sound" \
        --valid_data_path_and_name_and_type "data/dev_voc/wav.scp,speech_ref1,sound" \
        --train_shape_file "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic_again/train/speech_ref1_shape" \
        --valid_shape_file "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic_again/valid/speech_ref1_shape" \
        --ngpu 2 --multiprocessing_distributed true --unused_parameters true \
        --generator_first false \
        --resume true 
fi
# # ─────────────────────────────────────────────────────────────────────────
# # Stage 10: Inference
# # ─────────────────────────────────────────────────────────────────────────
# if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
#     log "Stage 10: Inference"
#     for tset in ${test_sets}; do
#         log "  ${tset}"
#         ${python} local/upsample_wav_scp.py \
#             --wav_scp   data/libritts_${tset}/wav.scp \
#             --out_dir   data/libritts_${tset}_16k \
#             --target_sr 16000 \
#             --nj        ${nj}

#         ${python} -m espnet2.bin.enh_inference_speech_cleaner \
#             --fp_train_config  "${fp_exp}/config.yaml" \
#             --fp_model_file    "${fp_exp}/valid.loss.best.pth" \
#             --voc_train_config "${voc_finetune_exp}/config.yaml" \
#             --voc_model_file   "${voc_finetune_exp}/valid.loss_G.best.pth" \
#             --wav_scp          "data/libritts_${tset}_16k/wav.scp" \
#             --output_dir       "${restored_dir}/${tset}" \
#             --batch_size 1 --device cuda --dtype bfloat16
#     done
# fi
# ─────────────────────────────────────────────────────────────────────────
# Stage 10: Inference
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    log "Stage 10: Inference"
    for tset in ${test_sets}; do
        log "  ${tset}"
        ${python} -m espnet2.bin.enh_inference_speech_cleaner \
            --fp_train_config  "exp/speech_cleaner_fp_xeus_single/config.yaml" \
            --fp_model_file    "exp/speech_cleaner_fp_xeus_single/valid.loss.best.pth" \
            --voc_train_config "exp/speech_cleaner_voc_finetune_xeus_single_a100_dnsmos_new_loss/config.yaml" \
            --voc_model_file   "exp/speech_cleaner_voc_finetune_xeus_single_a100_dnsmos_new_loss/51epoch.pth" \
            --wav_scp          "data/libritts_${tset}_16k/wav.scp" \
            --output_dir       "exp/restored_xeus_single_a100_dnsmos_new_loss-voc-epoch51/${tset}" \
            --batch_size 1 --device cuda --dtype bfloat16
    done
fi

# ─────────────────────────────────────────────────────────────────────────
# Stage 10: Inference
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Inference"
    for tset in ${test_sets}; do
        log "  ${tset}"

        ${python} -m espnet2.bin.enh_inference_speech_cleaner \
            --fp_train_config  "exp/speech_cleaner_fp_xeus_multi_all/config.yaml" \
            --fp_model_file    "exp/speech_cleaner_fp_xeus_multi_all/valid.loss.best.pth" \
            --voc_train_config "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic/config.yaml" \
            --voc_model_file   "exp/speech_cleaner_voc_finetune_xeus_multi_all_utterance_dynamic/valid.stoi.ave_5best.pth" \
            --wav_scp          "data/libritts_${tset}_16k/wav.scp" \
            --output_dir       "exp/restored_xeus_multi_all_utterance/stoi_avg_best5/${tset}" \
            --batch_size 1 --device cuda --dtype bfloat16
    done
fi



# ─────────────────────────────────────────────────────────────────────────
# Stage 11: Scoring
# ─────────────────────────────────────────────────────────────────────────
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "Stage 11: Scoring"
    for tset in ${test_sets}; do
        ${python} local/score_v2.py \
            --restored_scp  "exp/restored_xeus_single/${tset}/wav.scp" \
            --noisy_scp "data/libritts_${tset}/wav.scp" \
            --text "data/libritts_${tset}/text"\
            --out_dir    "exp/scores/xeus_single/${tset}" \
            --metrics "dnsmos" "nisqa" "spksim" "utmos" "squim_noref" \
            --asr_model "owsm-v3.1"\
            --device "cuda"
    done
fi

# # ─────────────────────────────────────────────────────────────────────────
# # Stage 11: Scoring
# # ─────────────────────────────────────────────────────────────────────────
# if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
#     log "Stage 11: Scoring"
#     for tset in ${test_sets}; do
#         ${python} local/score_v2.py \
#             --restored_scp  "${restored_dir}/${tset}/wav.scp" \
#             --noisy_scp "data/libritts_${tset}/wav.scp" \
#             --text "data/libritts_${tset}/text"\
#             --out_dir    "exp/scores/${tset}" \
#             --metrics "wer" "dnsmos" "nisqa" "spksim" "utmos" "squim_noref" \
#             --asr_model "owsm-v3.1"\
#             --device "cuda"
#     done
# fi

log "=== Done (stage ${stage} ~ ${stop_stage}) ==="