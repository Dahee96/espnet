#!/usr/bin/env python3
"""Training entry point for Speech Cleaner (speech restoration).

Usage
-----
# Stage 1 — Feature Predictor
python -m espnet2.bin.enh_train_speech_cleaner \
    --task sc_fp \
    --config conf/train_speech_cleaner_fp.yaml \
    --train_data_path_and_name_and_type \
        "data/train_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        "data/train_paired_16k/clean/wav.scp,speech_ref1,sound" \
    --valid_data_path_and_name_and_type \
        "data/dev_paired_16k/noisy/wav.scp,noisy_speech,sound" \
        "data/dev_paired_16k/clean/wav.scp,speech_ref1,sound" \
    --output_dir exp/speech_cleaner_fp \
    --ngpu 8

# Stage 2 — Vocoder pretrain (clean SSL features)
python -m espnet2.bin.enh_train_speech_cleaner \
    --task sc_gan \
    --config conf/train_speech_cleaner_voc_pretrain.yaml \
    --use_predicted_feat false \
    --output_dir exp/speech_cleaner_voc_pretrain \
    --ngpu 8

# Stage 3 — Vocoder finetune (predicted SSL features)
python -m espnet2.bin.enh_train_speech_cleaner \
    --task sc_gan \
    --config conf/train_speech_cleaner_voc_finetune.yaml \
    --use_predicted_feat true \
    --fp_model_path exp/speech_cleaner_fp/valid.loss.best.pth \
    --output_dir exp/speech_cleaner_voc_finetune \
    --ngpu 8
"""

import sys

from espnet2.tasks.speech_cleaner import (  # noqa: E402
    SpeechCleanerFPTask,
    SpeechCleanerGANTask,
)


def main(cmd=None):
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", required=True, choices=["sc_fp", "sc_gan"])
    pre_args, remaining = pre.parse_known_args(cmd)

    if pre_args.task == "sc_fp":
        SpeechCleanerFPTask.main(cmd=remaining)
    else:
        SpeechCleanerGANTask.main(cmd=remaining)


if __name__ == "__main__":
    main()
