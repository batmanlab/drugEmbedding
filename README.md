# Semi-Supervised Hierarchical Drug Embedding in Hyperbolic Space
![KDD_schematic_diagram_v5](https://user-images.githubusercontent.com/8482358/93242235-7cb87b80-f754-11ea-931d-03f92a940935.png)
This is the repository for the paper published in JCIM: \
[Semi-supervised Hierarchicical Drug Embedding in Hyperbolic Space](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00681)

## Abstract
Learning accurate drug representations is essential for tasks such as computational drug repositioning and prediction of drug side-effects. A drug hierarchy is a valuable source that encodes human knowledge of drug relations in a tree-like structure where drugs that act on the same organs, treat the same disease, or bind to the same biological target are grouped together. However, its utility in learning drug representations has not yet been explored, and currently described drug representations cannot place novel molecules in a drug hierarchy. 

Here, we develop a semi-supervised drug embedding that incorporates two sources of information: (1) underlying chemical grammar that is inferred from chemical structures of drugs and  drug-like molecules (unsupervised), and (2) hierarchical relations that are encoded in an expert-crafted hierarchy of approved drugs (supervised). We use the Variational Auto-Encoder (VAE) framework to encode the chemical structures of molecules and use the knowledge-based drug-drug similarity to induce the clustering of drugs in hyperbolic space. The hyperbolic space is amenable for encoding hierarchical relations. Both quantitative and qualitative results support that the learned drug embedding can accurately reproduce the chemical structure and induce the hierarchical relations among drugs. Furthermore, our approach can infer the pharmacological properties of novel molecules by retrieving similar drugs from the embedding space. We demonstrate that the learned drug embedding can be used to find new uses for existing drugs and to discover side-effects. We show that it significantly outperforms baselines in both tasks.

By
* Ke Yu
* Shyam Visweswaran
* Kayhan Batmanghelich

### [Bibtex](https://pubs.acs.org/action/showCitFormats?doi=10.1021%2Facs.jcim.0c00681&href=/doi/10.1021%2Facs.jcim.0c00681)
    @article{doi:10.1021/acs.jcim.0c00681,
    author = {Yu, Ke and Visweswaran, Shyam and Batmanghelich, Kayhan},
    title = {Semi-supervised Hierarchical Drug Embedding in Hyperbolic Space},
    journal = {Journal of Chemical Information and Modeling},
    volume = {60},
    number = {12},
    pages = {5647-5657},
    year = {2020},
    doi = {10.1021/acs.jcim.0c00681},
    note = {PMID: 33140969},
    URL = {https://doi.org/10.1021/acs.jcim.0c00681},
    eprint = {https://doi.org/10.1021/acs.jcim.0c00681}
    }

### To train a new model (shell script):
<pre><code>
#!/usr/bin/env bash

set -x

EXPERIMENT="my_model"
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
  --limit=0 \
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

# Python files:
* main.py: pipeline of the training procedure
* drugdata.py: dataloader of SMILES strings and the ATC hierarchy
* hvae.py: hyperbolic VAE functions
* lorentz.py: Lorentz model funtions
* decode.py: SMILES reconstruction function
* evae.py: classic VAE functions
* metrics.py: dendrogram purity score
* utils.py: utility functions

# Data:
* /DrugEmbedding/data/fda_drugs: SMILES strings and the ATC hierarchy table
* /DrugEmbedding/data/repoDB: repoDB dataset
* /DrugEmbedding/data/sider/deepchem: SIDER dataset
* /DrugEmbedding/data/pdbbind: PDBbind dataset
* /DrugEmbedding/data/tox21L: Tox21 dataset

# Notebooks:
Sample code used for experiments in the paper
