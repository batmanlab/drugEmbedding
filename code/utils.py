import os
import json
import torch
import numpy as np
from rdkit import Chem

def check_canonical_form(smiles):
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
    w2i = dict()
    i2w = dict()

    # add special tokens to vocab
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    for st in special_tokens:
        i2w[len(w2i)] = st
        w2i[st] = len(w2i)

    # load unique chars
    char_list = json.load(open(os.path.join(directory, vocab_file)))
    for i, c in enumerate(char_list):
        i2w[len(w2i)] = c
        w2i[c] = len(w2i)

    return w2i, i2w


def to_cuda_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def epoch_update_metrics(summary_writer, split, epoch, loss, nll_loss, kl_loss, kl_weight, lr):
    # convert list to numpy array
    loss = np.asarray(loss)
    nll_loss = np.asarray(nll_loss)
    kl_loss = np.asarray(kl_loss)
    kl_weight = np.asarray(kl_weight)
    lr = np.asarray(lr)

    summary_writer.add_scalar('%s/avg_total_loss' % split, loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_nll_loss' % split, nll_loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_kl_loss' % split, kl_loss.mean(), epoch)
    summary_writer.add_scalar('%s/avg_kl_weight' % split, kl_weight.mean(), epoch)
    summary_writer.add_scalar('%s/avg_learning_rate' % split, lr.mean(), epoch)
    return


def batch_update_metrics(summary_writer, split, step, loss, nll_loss, kl_loss, kl_weight, lr):
    summary_writer.add_scalar('%s/total_loss' % split, loss, step)
    summary_writer.add_scalar('%s/null_loss' % split, nll_loss, step)
    summary_writer.add_scalar('%s/kl_loss' % split, kl_loss, step)
    summary_writer.add_scalar('%s/kl_weight' % split, kl_weight, step)
    summary_writer.add_scalar('%s/learaing_rate' % split, lr, step)
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