# Python files:
* main.py: pipeline of the training procedure
* drugdata.py: dataloader of SMILES strings and the ATC hierarchy
* hvae.py: hyperbolic VAE functions
* lorentz.py: Lorentz model funtions
* decode.py: SMILES reconstruction function
* evae.py: classic VAE functions
* metrics.py: dendrogram purity score
* utils.py: utility functions

# Shell script:
* run_pipeline.sh: define hyperparameters

# Experiments:
* kdd_010: checkpoint of the best performing model

# Data:
* /DrugEmbedding/data/fda_drugs: SMILES strings and the ATC hierarchy table
* /DrugEmbedding/data/repoDB: repoDB dataset
* /DrugEmbedding/data/sider/deepchem: SIDER dataset
* /DrugEmbedding/data/pdbbind: PDBbind dataset

