import os
import json
import numpy as np
from collections import OrderedDict
from scipy.cluster.hierarchy import linkage

from utils import *

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from evae import *
from hvae import *
from drugdata import *

from tqdm import tqdm
import itertools

def load_configs(experiment_dir, exp_name):
    experiment_path = os.path.join(experiment_dir, exp_name)

    # get experiment configurations
    with open(os.path.join(experiment_path, 'configs.json'), 'r') as fp:
        configs = json.load(fp)
    fp.close()
    return configs

def load_dataset(experiment_dir, exp_name, dataset_name):
    experiment_path = os.path.join(experiment_dir, exp_name)
    configs = load_configs(experiment_dir, exp_name)

    dataset = drugdata(task=configs['task'],
                       fda_drugs_dir=configs['data_dir'],
                       fda_smiles_file=configs['fda_file'],
                       fda_vocab_file=configs['vocab_file'],
                       fda_drugs_sp_file=configs['atc_sim_file'],
                       experiment_dir=experiment_path,
                       smi_file=dataset_name,
                       max_sequence_length=configs['max_sequence_length'],
                       fda_prop=configs['fda_prop'],
                       nneg=configs['nneg']
                       )
    return dataset

def load_model(experiment_dir, exp_name, checkpoint):
    experiment_path = os.path.join(experiment_dir, exp_name)
    configs = load_configs(experiment_dir, exp_name)

    dataset = drugdata(task=configs['task'],
                       fda_drugs_dir=configs['data_dir'],
                       fda_smiles_file=configs['fda_file'],
                       fda_vocab_file=configs['vocab_file'],
                       fda_drugs_sp_file=configs['atc_sim_file'],
                       experiment_dir=experiment_path,
                       smi_file='smiles_train.smi',
                       max_sequence_length=configs['max_sequence_length'],
                       fda_prop=configs['fda_prop'],
                       nneg=configs['nneg']
                       )

    # load model
    if configs['manifold_type'] == 'Euclidean':
        model = EVAE(
            vocab_size=dataset.vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha']
        )
    elif configs['manifold_type'] == 'Lorentz':
        model = HVAE(
            vocab_size=dataset.vocab_size,
            hidden_size=configs['hidden_size'],
            latent_size=configs['latent_size'],
            bidirectional=configs['bidirectional'],
            num_layers=configs['num_layers'],
            word_dropout_rate=configs['word_dropout_rate'],
            max_sequence_length=configs['max_sequence_length'],
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            prior=configs['prior_type'],
            alpha=configs['alpha']
        )

    checkpoint_path = os.path.join(experiment_path, checkpoint)
    torch.no_grad()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# FDA drugs sampler
class FDASampler(Sampler):

    def __init__(self, fda_idx):
        self.fda_idx = fda_idx

    def __iter__(self):
        return iter(self.fda_idx)

    def __len__(self):
        return len(self.fda_idx)

def dendrogram_purity_score(experiment_dir, exp_name, dataset_name, checkpoint):
    # load configs, dataset and model
    configs = load_configs(experiment_dir, exp_name)
    dataset = load_dataset(experiment_dir, exp_name, dataset_name)
    model = load_model(experiment_dir, exp_name, checkpoint)

    # create train dataloader
    fda_dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=configs['batch_size'],
                        sampler=FDASampler(dataset.fda_idx),
                        pin_memory=torch.cuda.is_available()
                    )

    # read drug names, drug latent reps.
    mean_lst = []
    drug_lst = []
    for iteration, batch in enumerate(fda_dataloader):
        if configs['manifold_type'] == 'Euclidean':
            mean, logv, z = model.get_intermediates(batch)
        else:
            mean, logv, _, _, z = model.get_intermediates(batch)
        mean_lst = mean_lst + mean.tolist()
        drug_lst = drug_lst + batch['drug_name']

    # read ATC hierarchy
    df_drug_atc_path = drug_atc_path(drug_lst, atc_level=4)

    # compute linkage matrix
    fda_mean = torch.tensor(mean_lst)
    lor_dist_condensed = pairwise_dist(configs['manifold_type'], fda_mean.numpy())
    linkage_matrix = linkage(lor_dist_condensed, 'single')

    N = fda_mean.shape[0]
    LL = [[item] for item in range(N)]
    for j in range(linkage_matrix.shape[0]):
        p, q = int(linkage_matrix[j][0]), int(linkage_matrix[j][1])
        LL.append(LL[p] + LL[q])

    # iterate through all true cluster labels
    C = df_drug_atc_path['ATC_PATH'].unique()

    p_cnt = 0
    purity = 0
    for ck in C:
        drugs = df_drug_atc_path[df_drug_atc_path['ATC_PATH'] == ck]['ATC_LVL5'].unique()
        if len(drugs) > 1:
            drugs_pair = list(itertools.combinations(drugs, 2))  # list of drugs pair tuple
            for p in drugs_pair:
                drug_1_idx = drug_lst.index(p[0])
                drug_2_idx = drug_lst.index(p[1])
                lca = [item for item in LL if drug_1_idx in item and drug_2_idx in item][0]
                lvs = np.array(drug_lst)[lca]
                ck_predict = np.array(drug_atc_path(lvs.tolist(), atc_level=4)['ATC_PATH'])
                purity += sum(ck_predict == ck) / len(ck_predict)
                p_cnt += 1

    dp = purity / p_cnt
    return dp

experiment_dir = './experiments/SMILES'
exp_name = 'debug_dp'
dataset_name = 'smiles_test.smi'
checkpoint = 'checkpoint_epoch001_batch268.model'

dp = dendrogram_purity_score(experiment_dir, exp_name, dataset_name, checkpoint)


def purity_score(linkage_matrix, y_test, repeats, seed):
    """Compute the expected dendrogram purity.
    Sample a leaf uniformly at random. Then sample another leaf from the same
    true class uniformly at random. Find their lowest common ancestor in the
    tree and compute purity with respect to that class.
    return purity_score
    """
    np.random.seed(seed)
    purity = 0
    N = len(y_test)

    for i in range(repeats):
        class_test = np.random.choice(y_test, 1)[0]
        leaf1, leaf2 = np.random.choice(np.arange(N)[np.array(y_test) == class_test], size=2, replace=None)

        LL = [[item] for item in range(N)]
        for j in range(linkage_matrix.shape[0]):
            p, q = int(linkage_matrix[j][0]), int(linkage_matrix[j][1])
            LL.append(LL[p] + LL[q])
        common_ancestor = [item for item in LL if leaf1 in item and leaf2 in item][0]
        predict_label = np.array(y_test)[common_ancestor]

        purity += sum(predict_label == y_test[leaf1]) / len(predict_label)

    return purity / repeats


mdat = np.array([[ 0.93637874,  1.61258974],
       [ 1.95192875,  2.84452075],
       [ 2.07671748,  3.24442548],
       [ 3.122903  ,  4.516753  ],
       [ 3.56202194,  5.17531994],
       [ 3.53211875,  5.75857675],
       [ 4.65794237,  6.66995537],
       [ 5.83738797,  8.46562797],
       [ 6.22595817,  9.28082817],
       [ 6.51552067,  9.36110867],
       [ 7.24619975,  3.68958775],
       [ 6.50554148,  3.69771048],
       [ 6.58213752,  4.31283952],
       [ 6.02279742,  4.52753342],
       [ 5.83280398,  4.85751598],
       [ 5.12305078,  4.76874878],
       [ 5.0430706 ,  5.2911986 ],
       [ 2.44081699,  6.35402999]])

mdat_y = list(np.repeat('A',10))
mdat_y.extend(list(np.repeat('B',8)))

single_test = linkage(mdat, method='single')


dp = purity_score(single_test, mdat_y, 5, 12)


stop = 0




# load configs, dataset and model
experiment_dir = './experiments/SMILES'
exp_name = 'debug_dp'
dataset_name = 'smiles_test.smi'
checkpoint = 'checkpoint_epoch001_batch268.model'
configs = load_configs(experiment_dir, exp_name)
dataset = load_dataset(experiment_dir, exp_name, dataset_name)
model = load_model(experiment_dir, exp_name, checkpoint)

# create train dataloader
fda_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=configs['batch_size'],
                    sampler=FDASampler(dataset.fda_idx),
                    pin_memory=torch.cuda.is_available()
                )

# read drug names, drug latent reps.
mean_lst = []
drug_lst = []
for iteration, batch in tqdm(enumerate(fda_dataloader)):
    if configs['manifold_type'] == 'Euclidean':
        mean, logv, z = model.get_intermediates(batch)
    else:
        mean, logv, _, _, z = model.get_intermediates(batch)
    mean_lst = mean_lst + mean.tolist()
    drug_lst = drug_lst + batch['drug_name']

# read ATC hierarchy
df_drug_atc_path = drug_atc_path(drug_lst, atc_level=4)

# compute linkage matrix
fda_mean = torch.tensor(mean_lst)
lor_dist_condensed = pairwise_dist('Lorentz', fda_mean.numpy())
linkage_matrix = linkage(lor_dist_condensed, 'single')

N = fda_mean.shape[0]
LL = [[item] for item in range(N)]
for j in range(linkage_matrix.shape[0]):
    p, q = int(linkage_matrix[j][0]), int(linkage_matrix[j][1])
    LL.append(LL[p] + LL[q])

# iterate through all true cluster labels
C = df_drug_atc_path['ATC_PATH'].unique()

p_cnt = 0
purity = 0
for ck in C:
    drugs = df_drug_atc_path[df_drug_atc_path['ATC_PATH']==ck]['ATC_LVL5'].unique()
    if len(drugs) > 1:
        drugs_pair = list(itertools.combinations(drugs, 2)) # list of drugs pair tuple
        for dp in drugs_pair:
            drug_1_idx = drug_lst.index(dp[0])
            drug_2_idx = drug_lst.index(dp[1])
            lca = [item for item in LL if drug_1_idx in item and drug_2_idx in item][0]
            lvs = np.array(drug_lst)[lca]
            ck_predict = np.array(drug_atc_path(lvs.tolist(), atc_level=4)['ATC_PATH'])
            purity += sum(ck_predict == ck) / len(ck_predict)
            p_cnt += 1
dp = purity/p_cnt
stop = True