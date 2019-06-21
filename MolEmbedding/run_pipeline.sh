#!/bin/sh

set -x

EXPERIMENT="exp_b_3"
DATA_DIR="./data"
DATA_FILE="250k_rndm_zinc_drugs_clean.smi"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python main.py \
  --data_dir="${DATA_DIR}" \
  --data_file="${DATA_FILE}" \
  --vocab_file="zinc_char_list.json" \
  --checkpoint_dir="./experiments/SMILES" \
  --experiment_name="${EXPERIMENT}" \
  --limit=249456 \
  --batch_size=128 \
  --epochs=60 \
  --max_sequence_length=120 \
  --learning_rate=3e-4 \
  --max_norm=1e12 \
  --wd=0 \
  --manifold_type="Euclidean" \
  --rnn_type="gru" \
  --bidirectional=False \
  --num_layers=4 \
  --hidden_size=500 \
  --latent_size=200 \
  --one_hot_rep=True \
  --word_dropout_rate=0.2 \
  --embedding_dropout_rate=0.0 \
  --anneal_function="logistic" \
  --k=0.51 \
  --x0=29 \
  --num_workers=1 \
  --logging_steps=1 \
  --save_per_epochs=5 \
  --new_training=True \
  --new_annealing=False \
  --checkpoint="checkpoint_epoch060.model" \
  --trained_epochs=60 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1