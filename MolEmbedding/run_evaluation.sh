#!/bin/sh

set -x

EXPERIMENT="gpu_lor_20_250K"
CHECKPOINT="checkpoint_epoch100.model"
EVALUATION="${EXPERIMENT}_${CHECKPOINT}.eval"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python evaluation.py \
    --experiment_name="${EXPERIMENT}" \
    --checkpoint="${CHECKPOINT}" >> ${LOG_DIR}/${EVALUATION}.log 2>&1