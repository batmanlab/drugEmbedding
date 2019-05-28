#!/bin/sh

set -x

EXPERIMENT="gpu_lor_5_drugs"
DATA_DIR="./data"
DATA_FILE="all_drugs_only.smi"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python main.py \
  --data_dir="${DATA_DIR}" \
  --data_file="${DATA_FILE}" \
  --vocab_file="zinc_char_list.json" \
  --checkpoint_dir="./experiments" \
  --experiment_name="${EXPERIMENT}" \
  --limit=1385 \
  --batch_size=32 \
  --epochs=100 \
  --max_sequence_length=120 \
  --learning_rate=1e-3 \
  --manifold_type="Euclidean" \
  --rnn_type="gru" \
  --bidirectional=False \
  --num_layers=1 \
  --hidden_size=400 \
  --latent_size=5 \
  --one_hot_rep=True \
  --word_dropout_rate=0.25 \
  --embedding_dropout_rate=0.2 \
  --anneal_function="logistic" \
  --k=0.0025 \
  --x0=2500 \
  --num_workers=1 \
  --logging_steps=25 \
  --save_per_epochs=10 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1