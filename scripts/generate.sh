#!/bin/bash
set -eo pipefail

export CUDA_VISIBLE_DEVICES=0

fairseq-generate /data/hrsun/data/MUST-C/en-de \
    --user-dir /home/hrsun/Speech/MoyuST/MoyuST \
    --criterion multi_task_cross_entropy_force_alignment \
    --gen-subset tst-COMMON_st --task speech_to_text --prefix-size 1 \
    --quiet \
    --lenpen 0.7 \
    --batch-size 16 --max-source-positions 4000000 --beam 10 \
    --config-yaml config_st.yaml  --path /data/hrsun/models/SpeechTrans/MoyuNetForceV3_fix_full.pt \
    --scoring sacrebleu
