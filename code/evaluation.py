import os
import json
import pickle
from model import *
from collections import OrderedDict
from textdata import textdata
from utils import *
import pandas as pd
import torch
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
import matplotlib.pyplot as plt

def smile_generator(configs, model, nsample, z=None, sampling_mode='greedy'):

    w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])

    eos_idx = w2i['<eos>']

    samples_idx, z = model.inference(n=nsample, z=z, sampling_mode=sampling_mode)

    smiles_lst = []
    for i in range(nsample):
        smiles = []
        for j in range(configs['max_sequence_length']):
            if samples_idx[i,j] == eos_idx:
                break
            smiles.append(i2w[samples_idx[i,j].item()])
        smiles = "".join(smiles)
        smiles_lst.append(smiles)

    #for smi in smiles_lst:
        #try:
        #    cans = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
        #    print('%s is a valid smiles. Its canonoical form is %s'%(smi, cans))
        #except:
        #    print('%s is not a valid smiles'%smi)

    return smiles_lst


"""
experiment 1: sample a few SMILES from prior distribution P(z)
"""
def func_exp1(nsample):
    #inference sampling SMILES from random z
    nsample = 1000
    vae_smiles_sample = smile_generator(configs, model, nsample, sampling_mode='greedy')
    return vae_smiles_sample

"""
experiment 2: distribution of SMILES sampled from prior distribution P(z)
"""
def func_exp2(vae_smiles_sample):
    nsample = len(vae_smiles_sample)
    dataset_raw_file = os.path.join(configs['data_dir'], 'smiles_train.smi')
    smiles_train = []
    with open(dataset_raw_file, 'r') as file:
        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_train.append(''.join(words))
    file.close()

    #create SMILES target list (all 250K SMILES)
    dataset_raw_file = os.path.join(configs['data_dir'], '250k_rndm_zinc_drugs_clean.smi')
    smiles_target = []
    with open(dataset_raw_file, 'r') as file:
        for seq in file.readlines():
            words = list(seq)[:-1]
            smiles_target.append(''.join(words))
    file.close()


    unique_vae_smiles = list(set(vae_smiles_sample))
    smi_freq_lst = []
    for smi in unique_vae_smiles:
        # count of duplicate smiles
        smi_cnt = vae_smiles_sample.count(smi)

        # valid or invalide
        if type(Chem.MolFromSmiles(smi)) is Chem.rdchem.Mol:
            smi_valid = 1
        else:
            smi_valid = 0

        # exist in training data or not?
        if smi in smiles_train:
            in_train = 1
        else:
            in_train = 0

        # exist in the whole dataset or not?
        if smi in smiles_target:
            in_target = 1
        else:
            in_target = 0

        smi_freq_lst.append([smi, smi_cnt, smi_valid, in_train, in_target])
    smi_freq_df = pd.DataFrame(smi_freq_lst, columns=['VAE_SMILES', 'COUNT', 'VALID', 'IN_TRAIN', 'IN_TARGET'])

    #check validity
    print('%d out %d generated SMILES are unique.' % (len(unique_vae_smiles), nsample))
    print('%d VAE generated SMILES are valid molecules'%smi_freq_df['VALID'].sum())
    print('%d VAE generated SMILES are in training data'%smi_freq_df['IN_TRAIN'].sum())
    print('%d VAE generated SMILES are in the 250K drug-like SMILES dataset'%smi_freq_df['IN_TARGET'].sum())


"""
experiment 3: sample from posterior distribution
"""
def func_exp3(smile_x, model):
    nsample = 20

    input_sequence = [w2i['<sos>']]
    for i in smile_x:
        input_sequence.append(w2i[i])
    input_sequence.append(w2i['<eos>'])
    input_sequence = input_sequence + [0] * (configs['max_sequence_length'] - len(input_sequence)-1)
    input_sequence = np.asarray(input_sequence)
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(0)
    sequence_length = torch.tensor([len(smile_x)+1])

    if model.manifold_type == 'Euclidean':
        _, mean, logv, _= model.forward(input_sequence,sequence_length)
        std = torch.exp(0.5 * logv)
        z = to_cuda_var(torch.randn([nsample, mean.size(0)]))
        z = z * std + mean
    elif model.manifold_type == 'Lorentz':
        _, mean, logv, _ = model.forward(input_sequence, sequence_length)
        _, _, z = lorentz_sampling(mean.repeat(nsample,1), logv.repeat(nsample,1))

    smile_mean = smile_generator(configs, model, nsample=1, z=mean, sampling_mode='greedy')
    smiles_lst = smile_generator(configs, model, nsample, z=z, sampling_mode='greedy')

    smiles_grid = [smile_x] + smile_mean + smiles_lst

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
def func_exp4(fda_drugs_path, model):
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
        _, mean, logv, _ = model.forward(input_sequence, sequence_length)

        samples_idx, _ = model.inference(1, mean, sampling_mode='greedy')
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
experiment configuration
"""
exp_name = 'gpu_euc_5_drugs'
checkpoint = 'checkpoint_epoch100.model'
fda_drugs = 'all_drugs.smi'
exp_path = os.path.join('./experiments/' + exp_name, 'configs.json')
checkpoint_path = os.path.join('./experiments/' + exp_name, checkpoint)
img_path = os.path.join('./experiments/' + exp_name, checkpoint + '.png')
fda_drugs_path = os.path.join('./data/', fda_drugs)
embeddings_output_path = os.path.join('./experiments/'+exp_name, checkpoint + 'embeddings_output' + '.pickle')

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
    unk_idx=datasets['train'].unk_idx
)

# load checkpoint
model.load_state_dict(torch.load(checkpoint_path))
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

w2i, i2w = idx2word(configs['data_dir'], configs['vocab_file'])


"""
experiment 1: sample a few SMILES from prior distribution P(z)
"""
vae_smiles_sample = func_exp1(1000)


"""
experiment 2: distribution of SMILES sampled from prior distribution P(z)
"""
func_exp2(vae_smiles_sample)
"""
experiment 3: sample from posterior distribution
"""
smile_x = 'CC(C)Cc1ccc(C(C)C(=O)O)cc1' # Ibuprofen
func_exp3(smile_x, model)

"""
experiment 4: FDA drugs latent representations
"""
#drug_mu_dict = func_exp4(fda_drugs_path, model)

