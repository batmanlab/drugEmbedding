#!/usr/bin/env bash

set -x

EXPERIMENT="exp_002"
DATA_DIR="./data/fda_drugs"
DATA_FILE="smiles_set_clean.smi"
FDA_FILE="all_drugs.smi"
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

python main.py \
  --data_dir="${DATA_DIR}" \
  --data_file="${DATA_FILE}" \
  --fda_file="${FDA_FILE}" \
  --vocab_file="char_set_clean.pkl" \
  --atc_sim_file="drugs_sp_all.csv" \
  --checkpoint_dir="./experiments/SMILES" \
  --experiment_name="${EXPERIMENT}" \
  --task="vae" \
  --limit=0 \
  --batch_size=256 \
  --epochs=100 \
  --max_sequence_length=120 \
  --learning_rate=3e-4 \
  --max_norm=1e12 \
  --wd=0 \
  --manifold_type="Lorentz" \
  --prior_type="Standard" \
  --num_centroids=0 \
  --bidirectional=False \
  --num_layers=1 \
  --hidden_size=128 \
  --latent_size=2 \
  --word_dropout_rate=0.2 \
  --anneal_function="logistic" \
  --k=0.51 \
  --x0=29 \
  --C=0.0 \
  --num_workers=1 \
  --logging_steps=1 \
  --save_per_epochs=5 \
  --new_training=True \
  --new_annealing=True \
  --checkpoint="checkpoint_epoch100.model" \
  --trained_epochs=100 \
  --alpha=0.0 \
  --beta=0.5 \
  --nneg=5 \
  --fda_prop=0.2 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1
