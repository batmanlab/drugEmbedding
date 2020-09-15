# Semi-Supervised Hierarchical Drug Embedding in Hyperbolic Space
![TOC](https://user-images.githubusercontent.com/8482358/93241156-f8192d80-f752-11ea-89c3-ee0df1aae4de.png)
This is the repository for the manuscript:
[Semi-supervised Hierarchicical Drug Embedding in Hyperbolic Space](https://arxiv.org/abs/2006.00986)


### To train a new model (shell script):
<pre><code>
#!/usr/bin/env bash

set -x

EXPERIMENT="new_model"
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
  --checkpoint_dir="./experiments/EXP_TASK" \
  --experiment_name="${EXPERIMENT}" \
  --task="vae + atc" \
  --limit=5000 \
  --batch_size=128 \
  --epochs=200 \
  --max_sequence_length=120 \
  --learning_rate=3e-4 \
  --max_norm=1e12 \
  --wd=0 \
  --manifold_type="Lorentz" \
  --prior_type="Standard" \
  --num_centroids=0 \
  --bidirectional=False \
  --num_layers=1 \
  --hidden_size=512 \
  --latent_size=64 \
  --word_dropout_rate=0.2 \
  --anneal_function="logistic" \
  --k=0.51 \
  --x0=29 \
  --C=1.0 \
  --num_workers=4 \
  --logging_steps=1 \
  --save_per_epochs=5 \
  --new_training=False \
  --new_annealing=False \
  --checkpoint="checkpoint_epoch000.model" \
  --trained_epochs=00 \
  --alpha=0.0 \
  --beta=0.015625 \
  --gamma=0.0 \
  --delta=11.0 \
  --nneg=11 \
  --fda_prop=0.2 >> ${LOG_DIR}/${EXPERIMENT}.log 2>&1
</code></pre>
