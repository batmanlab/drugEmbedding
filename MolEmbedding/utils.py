import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit import rdBase
rdBase.DisableLog('rdApp.error') #disable RDKit warning messages
from rdkit.Chem import QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def mix_gaussian_sampling(pi_k, mu_k, logv_k, n):
    """
    sample n examples from the mixture Gaussian distribution
    :param pi_k: distribution of weights of k Gaussiaan distribution
    :param mu_k: means (k * d)
    :param logv_k: log diagnoal variance (k * d)
    :param n: number of samples
    :return:
    """
    return None

def smiles_to_tokens(data_dir, vocab_file, s):

    with open(os.path.join(data_dir, vocab_file), 'rb') as fp:
        char_list = pickle.load(fp)
    fp.close()

    s = s.strip()

    j = 0
    tokens_lst = []
    while j < len(s):
        # handle atoms with two characters
        if j < len(s) - 1 and s[j:j + 2] in char_list:
            token = s[j:j + 2]
            j = j + 2

        # handle atoms with one character including hydrogen
        elif s[j] in char_list:
            token = s[j]
            j = j + 1

        # handel unknown character
        else:
            token = '<unk>'
            j = j + 1

        tokens_lst.append(token)

    return tokens_lst


def create_smiles_lst(directory, datafile):
    """
    read lines of SMILES input file and output a list
    :param directory: directory of input file
    :param file: name of input file
    :return: list of SMILES
    """
    smiles_lst = []
    datapath = os.path.join(directory, datafile)
    with open(datapath, 'r') as f:
        for seq in f.readlines():
            words = seq.strip()
            smiles_lst.append(''.join(words))
    f.close()
    return smiles_lst


def check_canonical_form(smiles):
    """
    check if the input is canonical SMILES
    :param smiles: input list of SMILES
    :return: list of canonical SMILES, number of converted SMILES
    """
    cnt_cvrt_cans = 0
    for i in range(len(smiles)):
        smi = smiles[i]
        smi = smi[:-1] #remove new line char
        cans = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        if cans != smi:
            smiles[i] = cans + '\n'
            cnt_cvrt_cans = cnt_cvrt_cans + 1
    return smiles, cnt_cvrt_cans


def idx2word(directory, vocab_file):
    """
    create two dictionaries: word to index, index to word
    :param directory: directory of vocabulary file
    :param vocab_file: name of vocabulary file
    :return:
    """
    w2i = dict()
    i2w = dict()

    # add special tokens to vocab
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    for st in special_tokens:
        i2w[len(w2i)] = st
        w2i[st] = len(w2i)

    # load unique chars
    with open(os.path.join(directory, vocab_file), 'rb') as fp:
        char_list = pickle.load(fp)
    fp.close()
    #char_list = json.load(open(os.path.join(directory, vocab_file)))
    for i, c in enumerate(char_list):
        i2w[len(w2i)] = c
        w2i[c] = len(w2i)

    return w2i, i2w


def to_cuda_var(x):
    """
    convert torch CUP to GPU
    :param x: torch tensor
    :return: torch tensor
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def epoch_update_loss(summary_writer, split, epoch, loss, nll_loss, kl_loss, kl_weight, marginal_posterior_divergence):
    # convert list to numpy array
    loss = np.asarray(loss)
    nll_loss = np.asarray(nll_loss)
    kl_loss = np.asarray(kl_loss)
    kl_weight = np.asarray(kl_weight)
    marginal_posterior_divergence = np.asarray(marginal_posterior_divergence)

    summary_writer.add_scalar('%s/avg_total_loss' % split, loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_nll_loss' % split, nll_loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_kl_loss' % split, kl_loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_kl_weight' % split, kl_weight.mean(), epoch)
    summary_writer.add_scalar('%s/avg_marginal_posterior_divergence' % split, marginal_posterior_divergence.mean(), epoch)
    return


def all_drugs_smiles_only(fda_drugs_path):
    drugs_smi = []
    with open(fda_drugs_path, 'r') as file:
        for seq in file.readlines():
            if seq != '\n':
                line = seq.split(" ")
                drugs_smi.append(line[0])
    file.close()
    with open('./data/all_drugs_only.smi','a') as file:
        for smi in drugs_smi:
            file.write(smi+os.linesep)
    file.close()


def perturb_z(z, noise_norm, constant_norm=False):
    if noise_norm > 0.0:
        noise_vec = np.random.normal(0, 1, size=z.shape)
        noise_vec = noise_vec / np.linalg.norm(noise_vec, axis=1).reshape(z.shape[0], 1)
        if constant_norm:
            return z + (noise_norm * noise_vec)
        else:
            noise_amp = np.random.uniform(0, noise_norm, size=(z.shape[0],1))
            return z + (noise_amp * noise_vec)
    else:
        return z


def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None


def verify_chemical_validty(mol):
    try:
        Chem.SanitizeMol(mol)
        return 1
    except:
        pass
    return 0


def mol_sas(smi):
    mol = smiles_to_mol(smi)
    if mol is not None:
        sas = sascorer.calculateScore(mol)
        return sas
    else:
        return None


def mol_logp(smi):
    mol = smiles_to_mol(smi)
    if mol is not None:
        logP = MolLogP(mol)
        return logP
    else:
        return None


def mol_weight(smi):
    """
    :param smi: input SMILES
    :return: average molecular weight
    """
    mol = smiles_to_mol(smi)
    if mol is not None:
        MolWt = ExactMolWt(mol)
        return MolWt
    else:
        return None


def mol_qed(smi):
    mol = smiles_to_mol(smi)
    if mol is not None:
        qed = QED.qed(mol)
        return qed
    else:
        return None

def smiles2mean(configs, smile_x, model):
    """
    encode a SMILES to the posterior mean of q(z|x)
    :param configs: model configurations
    :param smile_x: an input SMILES string
    :param model: model checkpoint
    :return: the posterior mean in the latent space (a vector representation)
    """
    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])
    input_sequence = [w2i['<sos>']]
    tokens = smiles_to_tokens(configs['data_dir'], configs['vocab_file'], smile_x)
    for i in tokens:
        input_sequence.append(w2i[i])
    input_sequence.append(w2i['<eos>'])
    input_sequence = input_sequence + [0] * (configs['max_sequence_length'] - len(input_sequence)-1)
    input_sequence = np.asarray(input_sequence)
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
    sequence_length = torch.tensor([len(smile_x)+1])

    # run through encoder
    _, mean, logv, _, _, _ = model.forward(input_sequence, sequence_length)

    return mean, logv


def idx2smiles(configs, samples_idx):

    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])
    eos_idx = w2i['<eos>']

    nsamples = samples_idx.shape[0]
    smiles_lst = []
    for i in range(nsamples):
        smiles = []
        for j in range(configs['max_sequence_length']):
            if samples_idx[i,j] == eos_idx:
                break
            smiles.append(i2w[samples_idx[i,j].item()])
        smiles = "".join(smiles)
        smiles_lst.append(smiles)
    return smiles_lst


def latent2smiles(configs, model, z, nsamples, sampling_mode):
    """
    decode a point in latent space to a SMILES string
    :param configs: model configurations
    :param model: model checkpoint
    :param z: input latent point (a vector representation)
    :param nsample: number of generated SMILES (greedy = 1, beam search = beam width, random = nsample)
    :param sampling_mode: greedy, beam search or random (at the output softmax layer)
    :return: z, samples_idx, smiles_lst (input z, one-hot-vector index, a list of sampled SMILES)
    """
    # sample from posterior distribution
    if sampling_mode == 'greedy':
        samples_idx, z = model.inference(n=1, z=z, sampling_mode=sampling_mode)
    elif sampling_mode == 'beam':
        samples_idx, z = model.beam_search(z=z, B=nsamples)
    elif sampling_mode == 'random':
        samples_idx, z = model.inference(n=nsamples, z=z.repeat(nsamples,1), sampling_mode=sampling_mode)

    smiles_lst = idx2smiles(configs, samples_idx)
    return z, samples_idx, smiles_lst


def eval_prior_samples(configs, vae_smiles_sample):
    nsample = len(vae_smiles_sample)
    train_file = os.path.join(configs['checkpoint_dir'] + '/' + configs['experiment_name'], 'smiles_train.smi')
    smiles_train = []
    with open(train_file, 'r') as file:
        for seq in file.readlines():
            #words = list(seq)[:-1]
            words = smiles_to_tokens(configs['data_dir'], configs['vocab_file'], seq)
            smiles_train.append(''.join(words))
    file.close()

    unique_vae_smiles = list(set(vae_smiles_sample))
    smi_freq_lst = []
    for smi in unique_vae_smiles:
        # count of duplicate smiles
        smi_cnt = vae_smiles_sample.count(smi)

        mol = smiles_to_mol(smi)
        # valid or invalid SMILES
        if mol is not None and smi != '':
            smi_valid = 1
        else:
            smi_valid = 0

        # chemically valid or not
        chem_valid = verify_chemical_validty(mol)

        # exist in training data or not?
        if smi in smiles_train:
            in_train = 1
        else:
            in_train = 0

        smi_freq_lst.append([smi, smi_cnt, smi_valid, chem_valid, in_train])
    smi_freq_df = pd.DataFrame(smi_freq_lst, columns=['VAE_SMILES', 'COUNT', 'SMILES_VALID', 'CHEMICAL_VALID','IN_TRAIN'])

    #check validity
    perc_valid = (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum() / nsample
    perc_chem_valid = (smi_freq_df['COUNT'] * smi_freq_df['CHEMICAL_VALID']).sum() / nsample
    if (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum() > 0:
        perc_unique = smi_freq_df['SMILES_VALID'].sum() / (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum()
    else:
        perc_unique = 0.0
    if (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum() > 0:
        perc_novel = (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID'] * (1 - smi_freq_df['IN_TRAIN'])).sum() / (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum()
    else:
        perc_novel = 0.0

    return perc_valid, perc_chem_valid, perc_unique, perc_novel


def pairwise_dist(manifold_type, x):
    """
    pairwise distance, dim(x) = (n, d)
    :param manifold_type: manifold type, 'Euclidean' or 'Lorentz'
    :param x: input numpy array
    :return: pairwise distance matrix
    """
    if manifold_type == 'Lorentz':
        x0 = x[:,0].reshape(-1,1)
        x1 = x[:,1:]
        m = np.matmul(x1, x1.transpose()) - np.matmul(x0, x0.transpose())
        np.fill_diagonal(m, -1-1e-12)
        m = -m
        dm = np.log(m + np.sqrt(m ** 2 - 1))
        return upper_tri_indexing(dm) # convert to condense form
    elif manifold_type == 'Euclidean':
        dc = pdist(x, metric='euclidean')
        return dc

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]