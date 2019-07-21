import os
import json
import matplotlib.pyplot as plt

from lorentz_model import arccosh, lorentz_model

import pickle
from model import *
from collections import OrderedDict
from textdata import textdata
from utils import *
import pandas as pd
import torch


from rdkit import rdBase
rdBase.DisableLog('rdApp.error') #disable RDKit warning messages
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem import QED
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from absl import app
from absl import flags
from absl import logging




def dist_z(manifold_type, z1, z2):
    if manifold_type == 'Euclidean':
        d2 = torch.dist(z1, z2).item()
    elif manifold_type == 'Lorentz' :
        if lorentz_product(z1, z2).item() >= -1:
            d2 = 0
        else:
            d2 = arccosh(-lorentz_product(z1,z2)).item()
    return d2

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

def smile_generator(configs, model, nsample, z=None, sampling_mode='greedy'):

    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])

    eos_idx = w2i['<eos>']

    if sampling_mode != 'beam':
        samples_idx, z = model.inference(n=nsample, z=z, sampling_mode=sampling_mode)
    elif sampling_mode == 'beam':
        samples_idx, z = model.beam_search(z, B=nsample)

    smiles_lst = []
    for i in range(nsample):
        smiles = []
        for j in range(configs['max_sequence_length']):
            if samples_idx[i,j] == eos_idx:
                break
            smiles.append(i2w[samples_idx[i,j].item()])
        smiles = "".join(smiles)
        smiles_lst.append(smiles)
    return z, samples_idx, smiles_lst

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

def mol_logp(smi):
    mol = smiles_to_mol(smi)
    if mol is not None:
        logP = MolLogP(mol)
        return logP
    else:
        return None

def mol_sas(smi):
    mol = smiles_to_mol(smi)
    if mol is not None:
        sas = sascorer.calculateScore(mol)
        return sas
    else:
        return None

"""
experiment 1: sample a few SMILES from prior distribution P(z)
"""
def func_exp1(configs, model, nsample, sampling_mode):
    samples_idx, z = model.inference(n=nsample, sampling_mode=sampling_mode, z=None)
    vae_smiles_sample = idx2smiles(configs, samples_idx)
    return vae_smiles_sample

"""
experiment 2: distribution of SMILES sampled from prior distribution P(z)
"""
def func_exp2(configs, vae_smiles_sample):
    nsample = len(vae_smiles_sample)
    train_file = os.path.join(configs['checkpoint_dir'] + '/' + configs['experiment_name'], 'smiles_train.smi')
    smiles_train = []
    with open(train_file, 'r') as file:
        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_train.append(''.join(words))
    file.close()

    unique_vae_smiles = list(set(vae_smiles_sample))
    smi_freq_lst = []
    for smi in unique_vae_smiles:
        # count of duplicate smiles
        smi_cnt = vae_smiles_sample.count(smi)

        mol = smiles_to_mol(smi)
        # valid or invalid SMILES
        if mol is not None:
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
    perc_unique = smi_freq_df['SMILES_VALID'].sum() / (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum()
    perc_novel = (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID'] * (1 - smi_freq_df['IN_TRAIN'])).sum() / (smi_freq_df['COUNT'] * smi_freq_df['SMILES_VALID']).sum()


    logging.info('%f sampled SMILES from prior distribution are valid.' %perc_valid)
    logging.info('%f sampled SMILES from prior distribution are chemically valid.' %perc_chem_valid)
    logging.info('%f sampled SMILES from prior distribution are unique.' %perc_unique)
    logging.info('%f sampled SMILES from prior distribution are novel (not in training dataset)' %perc_novel)

    return perc_valid, perc_chem_valid, perc_unique, perc_novel

"""
experiment 3: sample from posterior distribution
"""
def func_exp3(configs, img_path, smile_x, model, sampling_mode):

    mean, _ = smiles2mean(configs, smile_x, model)
    z, samples_idx, smiles_lst = latent2smiles(configs, model, mean, nsamples=30, sampling_mode='beam')

    smiles_grid = [smile_x] + smiles_lst

    mol_grid=[]
    for smi in smiles_grid:
        try:
            mol_grid.append(Chem.MolFromSmiles(smi))
        except:
            pass

    img=Chem.Draw.MolsToGridImage(mol_grid,molsPerRow=5,subImgSize=(400,400), legends=smiles_grid)
    img.save(img_path)

"""
experiment 4: FDA drugs latent representations
"""
def func_exp4(configs, w2i, i2w, embeddings_output_path, fda_drugs_path, model, sampling_mode):
    drugs_dict = {}
    with open(fda_drugs_path, 'r') as file:
        for seq in file.readlines():
            if seq != '\n':
                line = seq.split(" ")
                drugs_dict[line[1]] = line[0]
    file.close()

    drugs_mu_dict = {}
    for key in drugs_dict.keys():
        smile_1st = drugs_dict[key]
        input_sequence = [w2i['<sos>']]
        for i in smile_1st:
            try:
                input_sequence.append(w2i[i])
            except:
                input_sequence.append(w2i['<unk>'])
        input_sequence.append(w2i['<eos>'])
        input_sequence = input_sequence + [0] * (configs['max_sequence_length'] - len(input_sequence) - 1)
        input_sequence = np.asarray(input_sequence)
        input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
        sequence_length = torch.tensor([len(smile_1st) + 1])
        _, mean, logv, _, _, _ = model.forward(input_sequence, sequence_length)

        samples_idx, _ = model.inference(1, mean, sampling_mode= sampling_mode)
        for i in range(1):
            smiles = []
            for j in range(configs['max_sequence_length']):
                if samples_idx[i, j] == w2i['<eos>']:
                    break
                smiles.append(i2w[samples_idx[i, j].item()])
            smiles = "".join(smiles)

        drugs_mu_dict[key] = [smile_1st, smiles, mean.cpu().detach().numpy()]

    pickle_out = open(embeddings_output_path,'wb')
    pickle.dump(drugs_mu_dict, pickle_out)
    pickle_out.close()
    return drugs_mu_dict

"""
experiment 5: reconstruction from posterior mean
"""
def func_exp5(configs, smile_x, model, attempts):

    mean, _ = smiles2mean(configs, smile_x, model)
    z, samples_idx, smiles_lst = latent2smiles(configs, model, mean, nsamples=attempts, sampling_mode='beam')

    return (smile_x in smiles_lst)


"""
experiment 6: reconstruction from neighbors of posterior mean
"""

def func_exp6(configs, w2i, smile_x, model, attempts, sampling_mode, noise_norm=0):

    """
    input_sequence = [w2i['<sos>']]
    for i in smile_x:
        input_sequence.append(w2i[i])
    input_sequence.append(w2i['<eos>'])
    input_sequence = input_sequence + [0] * (configs['max_sequence_length'] - len(input_sequence)-1)
    input_sequence = np.asarray(input_sequence)
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
    sequence_length = torch.tensor([len(smile_x)+1])

    if model.manifold_type == 'Euclidean':
        _, mean, logv, _, _, _= model.forward(input_sequence,sequence_length)
    elif model.manifold_type == 'Lorentz':
        _, mean, logv, _, _, _ = model.forward(input_sequence, sequence_length)
    """

    mean, _ = smiles2mean(configs, w2i, smile_x, model)
    if torch.cuda.is_available():
        mean = mean.cpu()

    z = mean.detach().numpy()
    z = np.tile(z, (attempts, 1))
    z = perturb_z(z, noise_norm)
    z = torch.from_numpy(z)
    z = z.float()
    smiles_lst = smile_generator(configs, model, attempts, z=to_cuda_var(z), sampling_mode = sampling_mode)

    # compute:
    # 1, whether smile_x is reconstructed from the attempts
    recon_flag = smile_x in smiles_lst
    # 2, what is the unique number of sampled smiles
    num_unique_smiles = len(list(set(smiles_lst)))
    # 3, what is the average z-distance of re-inputing them back to the encoder
    z_mean = np.tile(mean.detach().numpy(), (attempts,1))
    z_sample = z.detach().numpy()
    z_dist_sample = np.linalg.norm(z_sample-z_mean,axis=1).mean()

    z_resample_lst=[]
    for smi in smiles_lst:
        z_resample, _ = smiles2mean(configs, w2i, smi, model)
        z_resample_lst.append(z_resample.detach().numpy())
    z_resample = np.asarray(z_resample_lst).reshape(z_sample.shape)
    z_dist_resample = np.linalg.norm(z_resample-z_mean,axis=1).mean()

    #return recon_flag, num_unique_smiles, z_dist_sample, z_dist_resample

    return recon_flag

"""
experiment 7: statistics of randomly generated SMILES from prior distribution
"""
def func_exp7(vae_smiles_sample):
    smi_prop_lst = []
    for smi in vae_smiles_sample:
        mol = smiles_to_mol(smi)
        validity = verify_chemical_validty(mol)
        smi_len = len(smi)
        wt = mol_weight(smi)
        qed = mol_qed(smi)
        sas = mol_sas(smi)
        logP = mol_logp(smi)
        smi_prop_lst.append([smi, validity, smi_len, wt, qed, sas, logP])
    smi_prop_df = pd.DataFrame(smi_prop_lst, columns=['SMILES', 'VALIDITY', 'LENGTH', 'MolWeight', 'QED', 'SAS', 'logP'])
    return smi_prop_df

"""
experiment configuration
"""
FLAGS = flags.FLAGS
flags.DEFINE_string('experiment_name', 'cpu_lor_5_50K', 'Experiment name')
flags.DEFINE_string('checkpoint', 'checkpoint_epoch020.model', 'Checkpoint name')

def main(argv):
    del argv

    exp_name = FLAGS.experiment_name
    checkpoint = FLAGS.checkpoint

    fda_drugs = 'all_drugs.smi'
    exp_dir = './experiments/'
    exp_path = os.path.join(exp_dir + exp_name, 'configs.json')
    checkpoint_path = os.path.join(exp_dir + exp_name, checkpoint)

    train_file = os.path.join(exp_dir + exp_name, 'smiles_train.smi')
    valid_file = os.path.join(exp_dir + exp_name, 'smiles_valid.smi')
    test_file = os.path.join(exp_dir + exp_name, 'smiles_test.smi')

    img_path = os.path.join(exp_dir + exp_name, checkpoint + '.png')
    fda_drugs_path = os.path.join('./data/', fda_drugs)
    embeddings_output_path = os.path.join(exp_dir + exp_name, checkpoint + 'embeddings_output' + '.pickle')

    nsample = 1000
    attempts = 20


    # load configuration file
    with open(exp_path, 'r') as fp:
        configs = json.load(fp)

    # prepare train, valid, test datasets
    datasets = OrderedDict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        datasets[split] = textdata(directory=configs['data_dir'], split=split, vocab_file=configs['vocab_file'],
                                   max_sequence_length=configs['max_sequence_length'])
    # build model
    model = MolVAE(
        vocab_size=datasets['train'].vocab_size,
        embedding_size=datasets['train'].vocab_size,
        hidden_size=configs['hidden_size'],
        latent_size=configs['latent_size'],
        manifold_type=configs['manifold_type'],
        rnn_type=configs['rnn_type'],
        bidirectional=configs['bidirectional'],
        num_layers=configs['num_layers'],
        word_dropout_rate=configs['word_dropout_rate'],
        embedding_dropout_rate=configs['embedding_dropout_rate'],
        one_hot_rep=configs['one_hot_rep'],
        max_sequence_length=configs['max_sequence_length'],
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        prior_var=1.0
    )

    # load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    """
    experiment 10: Posterior sampling metrics
    """



    nsamples = 24
    smile_x = 'CC(C)[C@@H](CO)Nc1ccc(N)cc1Br'
    mean, logv = smiles2mean(configs, smile_x, model)
    std = torch.exp(0.5 * logv)
    if configs['manifold_type'] == 'Euclidean':
        z = to_cuda_var(torch.randn([nsamples, configs['latent_size']]))
        z = z * std + mean
        vt = None
        u = None
    elif configs['manifold_type'] == 'Lorentz':
        vt, u, z = lorentz_sampling(mean.repeat(nsamples, 1), logv.repeat(nsamples, 1))

    samples_idx, z = model.inference(n=z.shape[0], sampling_mode='greedy', z=z)
    smiles_x_samples = idx2smiles(configs, samples_idx)

    # target metrics
    mol_t = smiles_to_mol(smile_x)
    validity_t = verify_chemical_validty(mol_t)
    wt_t = mol_weight(smile_x)
    qed_t = mol_qed(smile_x)
    sas_t = mol_sas(smile_x)
    logP_t = mol_logp(smile_x)

    # samples metrics
    smi_smpl_lst = []
    for smi in smiles_x_samples:
        try:
            mol = smiles_to_mol(smi)
            validity = verify_chemical_validty(mol)
            wt = mol_weight(smi)
            qed = mol_qed(smi)
            sas = mol_sas(smi)
            logP = mol_logp(smi)
            smi_smpl_lst.append([smi, validity, wt, qed, sas, logP])
        except:
            pass
    smi_smpl_df = pd.DataFrame(smi_smpl_lst,
                               columns=['SMILES', 'VALIDITY', 'MolWeight', 'QED', 'SAS', 'logP'])


    stop = True

    """
    experiment 1: sample a few SMILES from prior distribution P(z)
    """
    vae_smiles_sample = func_exp1(configs, model, nsample, sampling_mode='greedy')

    """
    experiment 2: quality of SMILES sampled from prior distribution P(z)
    """
    perc_valid, perc_chem_valid, perc_unique, perc_novel = func_exp2(configs, vae_smiles_sample)


    """
    experiment 3: visualize 2D molecules from posterior distribution
    """
    #smile_x = 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5' # Morphine
    smile_x = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1' # Ibuprofen
    #smile_x = 'CSCC(=O)NNC(=O)c1c(C)oc(C)c1C'
    #smile_x = 'CC(C)[C@@H](CO)Nc1ccc(N)cc1Br'
    #smile_x = 'Nc1ncnc2nc[nH]c12'
    #smile_x = 'O=C(O)CCCCCCCC(=O)O'
    func_exp3(configs, img_path, smile_x, model, sampling_mode='greedy')

    """
    experiment 4: output FDA drugs latent representations
    
    drug_mu_dict = func_exp4(configs, w2i, i2w, embeddings_output_path, fda_drugs_path, model, sampling_mode='greedy')
    """

    """
    experiment 5: reconstruction from the mean of posterior distribution
    
    smiles_lst = []
    file = './data/all_drugs_only.smi'
    with open(file, 'r') as file:
gatechke        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_lst.append(''.join(words))
    file.close()

    success = 0
    count = 0
    recon_success = []
    for smile_x in smiles_lst:
        count += 1
        try:
            if func_exp5(configs, smile_x, model, attempts=attempts):
                recon_success.append([smile_x, 1])
                success += 1
            else:
                recon_success.append([smile_x, 0])
            logging.info('%d out of %d SMILES are correctly reconstructed within %d attempts using posterior mean point' %(success, count, attempts))
        except:
            pass
    """

    """
    experiment 6: reconstruction from the latent space around the point
    
    smiles_lst = []
    file = test_file
    with open(file, 'r') as file:
        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_lst.append(''.join(words))
    file.close()

    success = 0
    count = 0
    recon_success = []
    for smile_x in smiles_lst[:100]:
        count += 1
        if func_exp6(configs, w2i, smile_x, model, attempts=attempts, sampling_mode='greedy', noise_norm=1):
            recon_success.append([smile_x],1)
            success += 1
        else:
            recon_success.append([smile_x, 0])
        logging.info('%d out of %d SMILES are correctly reconstructed within %d attempts by searching around posterior mean point' % (success, count, attempts))
    """

    """
    experiment 7: Distributions of molecular properties (ZINC, FDA Drugs)
    smi_prop_lst = []
    file_path = os.path.join('./data/', '250k_rndm_zinc_drugs_clean.smi')
    with open(file_path, 'r') as f:
        for seq in f.readlines():
            if smiles_to_mol(seq) is not None:
                smi_len = len(seq)
                wt = mol_weight(seq)
                qed = mol_qed(seq)
                sas = mol_sas(seq)
                logP = mol_logp(seq)
                smi_prop_lst.append([seq, smi_len, wt, qed, sas, logP])

    smi_prop_df = pd.DataFrame(smi_prop_lst,
                               columns=['SMILES', 'LENGTH', 'MolWeight', 'QED', 'SAS',
                                        'logP'])
    smi_prop_df.to_csv(file_path + '.csv')
    zinc_df = pd.read_csv('./data/250k_rndm_zinc_drugs_clean.smi.csv')
    

    smiles_lst = []
    file = './data/all_drugs_only.smi'
    with open(file, 'r') as file:
        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_lst.append(''.join(words))
    file.close()
    #smi_prop_df_train = func_exp7(smiles_lst)
    smi_prop_df_vae = func_exp7(vae_smiles_sample)

    #c1 = smi_prop_df_train.mean()
    c2 = smi_prop_df_vae.mean()
    #c3 = smi_prop_df_train.std()
    c4 = smi_prop_df_vae.std()
    #df_rprt = pd.concat([c1,c2,c3,c4], axis=1)
    #df_rprt.columns = ['Train Mean','Sample Mean','Train STD','Sample STD']
    df_rprt = pd.concat([c2, c4])
    df_rprt.columns = ['Sample Mean', 'Sample STD']
    print(df_rprt)
    """

    """
    experiment 8: Molecular properties of sampled SMILES
    """
    smi_prop_lst = []
    vae_smiles_sample = func_exp1(configs, model, nsample, sampling_mode='greedy')
    for smi in vae_smiles_sample:
        try:
            mol = smiles_to_mol(smi)
            validity = verify_chemical_validty(mol)
            smi_len = len(smi)
            wt = mol_weight(smi)
            qed = mol_qed(smi)
            sas = mol_sas(smi)
            logP = mol_logp(smi)
            smi_prop_lst.append([smi, validity, smi_len, wt, qed, sas, logP])
        except:
            pass
    smi_prop_df = pd.DataFrame(smi_prop_lst, columns=['SMILES', 'VALIDITY', 'LENGTH', 'MolWeight', 'QED', 'SAS', 'logP'])
    print('Mean molecule weight: %f' %(smi_prop_df['MolWeight'].mean()))
    print('STD molecule weight: %f' %(smi_prop_df['MolWeight'].std()))
    print('Mean molecule QED: %f' %(smi_prop_df['QED'].mean()))
    print('STD molecule QED: %f' %(smi_prop_df['QED'].std()))
    print('Mean molecule SAS: %f' %(smi_prop_df['SAS'].mean()))
    print('STD molecule SAS: %f' %(smi_prop_df['SAS'].std()))
    print('Mean molecule logP: %f' %(smi_prop_df['logP'].mean()))
    print('STD molecule logP: %f' %(smi_prop_df['logP'].std()))


    """
    experiment 9: Pairwise distance between latent points of molecules
    """
    smiles_fda = create_smiles_lst('./data', 'all_drugs_only.smi')


# start training
if __name__ == '__main__':
    app.run(main)