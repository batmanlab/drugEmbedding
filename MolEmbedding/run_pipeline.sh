#!/bin/sh

set -x

EXPERIMENT="zinc_fda_lor_standard_kl"
DATA_DIR="./data/zinc_fda_drugs_clean"
DATA_FILE="smiles_set_clean.smi"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python main.py \
  --data_dir="${DATA_DIR}" \
  --data_file="${DATA_FILE}" \
  --vocab_file="char_set_clean.pkl" \
  --checkpoint_dir="./experiments/SMILES" \
  --experiment_name="${EXPERIMENT}" \
  --limit=250833 \
  --batch_size=128 \
  --epochs=100 \
  --max_sequence_length=120 \
  --learning_rate=3e-4 \
  --max_norm=1e12 \
  --wd=0 \
  --manifold_type="Lorentz" \
  --prior_type="Standard" \
  --num_centroids=20 \
  --rnn_type="gru" \
  --bidirectional=False \
  --num_layers=1 \
  --hidden_size=512 \
  --latent_size=40 \
  --one_hot_rep=True \
  --word_dropout_rate=0.2 \
  --embedding_dropout_rate=0.0 \
  --anneal_function="logistic" \
  --k=0.51 \
  --x0=29 \
  --C=1 \
  --num_workers=1 \
  --logging_steps=1 \
  --save_per_epochs=1 \
  --new_training=False \
  --new_annealing=True \
  --checkpoint="checkpoint_epoch100.model" \
  --trained_epochs=100 \
  --prior_var=1.0 \
  --beta=1.0 \
  --alpha=0.0 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1