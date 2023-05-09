from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
#from numba import jit

from DrugEmbedding.utils import *
from DrugEmbedding.evae import *
from DrugEmbedding.hvae import *
from DrugEmbedding.drugdata import *
import itertools
from tqdm import tqdm

# reproducibility
#torch.manual_seed(216)
#np.random.seed(216)
import json

def load_model(configs):
    dataset = drugdata(task=configs['task'],
                       fda_drugs_dir=configs['data_dir'],
                       fda_smiles_file=configs['fda_file'],
                       fda_vocab_file=configs['vocab_file'],
                       fda_drugs_sp_file=configs['atc_sim_file'],
                       experiment_dir=os.path.join(configs['checkpoint_dir'], configs['experiment_name']),
                       smi_file='smiles_train.smi',
                       max_sequence_length=configs['max_sequence_length'],
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
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
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
            alpha=configs['alpha'],
            beta=configs['beta'],
            gamma=configs['gamma']
        )

    del dataset

    torch.no_grad()
    checkpoint_path = os.path.join(configs['checkpoint_dir'] + '/' + configs['experiment_name'], configs['checkpoint'])
    model.load_state_dict(torch.load(checkpoint_path))
    if torch.cuda.is_available():
        model = model.cuda()

    return model


# FDA drugs sampler
class FDASampler(Sampler):

    def __init__(self, fda_idx):
        self.fda_idx = fda_idx

    def __iter__(self):
        return iter(self.fda_idx)

    def __len__(self):
        return len(self.fda_idx)

def fda_drug_rep(configs, dataset, model, all_drugs):
    # create train dataloader
    if all_drugs:  # dataset is loaded from 'all_drugs.smi'
        fda_dataloader = DataLoader(
            dataset=dataset,
            batch_size=configs['batch_size'],
            pin_memory=torch.cuda.is_available()
        )
    else:
        fda_dataloader = DataLoader(
            dataset=dataset,
            batch_size=configs['batch_size'],
            sampler=FDASampler(dataset.fda_idx),
            pin_memory=torch.cuda.is_available()
        )

    # read drug names, drug latent reps.
    smiles_lst = []
    mean_lst = []
    logv_lst = []
    drug_lst = []
    for iteration, batch in enumerate(tqdm(fda_dataloader)):
        if configs['manifold_type'] == 'Euclidean':
            mean, logv, z = model.get_intermediates(batch)
        else:
            mean, logv, _, _, z = model.get_intermediates(batch)
        mean_lst = mean_lst + mean.tolist()
        logv_lst = logv_lst + logv.tolist()
        drug_lst = drug_lst + batch['drug_name']
    return drug_lst, mean_lst, logv_lst

#@jit(nopython=True)
def dendrogram_purity_score(configs, drug_lst, mean_lst, atc_lvl):
    # read ATC hierarchy
    df_drug_atc_path = drug_atc_path(drug_lst, atc_lvl)

    # compute linkage matrix
    fda_mean = torch.tensor(mean_lst)
    dist_condensed = pairwise_dist(configs['manifold_type'], fda_mean.numpy())
    linkage_matrix = linkage(dist_condensed, 'complete')

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
                ck_predict = np.array(drug_atc_path(lvs.tolist(), atc_lvl)['ATC_PATH'])
                purity += sum(ck_predict == ck) / len(ck_predict)
                p_cnt += 1
    if p_cnt > 0:
        return purity / p_cnt
    else:
        return 0