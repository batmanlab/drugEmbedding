import torch
import pandas as pd
import numpy as np

def to_cuda_var(x):
    """
    convert torch CUP to GPU
    :param x: torch tensor
    :return: torch tensor
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_atc_hierarchy(drug_key):
    """
    :param drug_key: drug name
    :return: drug ATC hierarchy
    """
    drug_dict = {}
    df_atc = pd.read_csv('./data/fda_drugs/atc_fda_unique.csv', index_col=0) # load ATC hierarchy inf.
    drug_dict['drug_name'] = drug_key
    drug_dict['ATC_LVL4'] = df_atc[df_atc['ATC_LVL5'] == drug_key]['ATC_LVL4'].values[0]
    drug_dict['ATC_LVL3'] = df_atc[df_atc['ATC_LVL5'] == drug_key]['ATC_LVL3'].values[0]
    drug_dict['ATC_LVL2'] = df_atc[df_atc['ATC_LVL5'] == drug_key]['ATC_LVL2'].values[0]
    drug_dict['ATC_LVL1'] = df_atc[df_atc['ATC_LVL5'] == drug_key]['ATC_LVL1'].values[0]

    return drug_dict


def kl_anneal_function(configs, start_epoch, epoch):
    """
    KL annealing function
    :param configs: experiment configurations
    :param start_epoch: starting epoch
    :param epoch: current epoch
    :return: annealing weight
    """
    k = configs['k'] # logistic function rate
    x0 = configs['x0']
    C = configs['C'] # constant

    if configs['new_training'] is False and configs['new_annealing'] is True:
        epoch = epoch - start_epoch + 1

    if configs['anneal_function'] == 'logistic':
        return float(1 / (1 + np.exp(-k * (epoch - x0))))
    elif configs['anneal_function'] == 'linear':
        return min(1, epoch / x0)
    elif configs['anneal_function'] == 'constant':
        return C
    elif configs['anneal_function'] == 'stepwise':
        KL_weights = np.linspace(0.0, configs['beta'], 10).repeat(10)
        return KL_weights[epoch-1]

def track_gradient_norm(model):

    # check the norm of gradients
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_grad_norm = total_norm ** (1. / 2)

    enc_grad_norm = 0
    for p in model.encoder_rnn.parameters():
        param_norm = p.grad.data.norm(2)
        enc_grad_norm += param_norm.item() ** 2
    enc_grad_norm = enc_grad_norm ** (1. / 2)

    dec_grad_norm = 0
    for p in model.decoder_rnn.parameters():
        param_norm = p.grad.data.norm(2)
        dec_grad_norm += param_norm.item() ** 2
    dec_grad_norm = dec_grad_norm ** (1. / 2)

    h2m_wght_norm = 0
    h2m_grad_norm = 0
    for p in model.hidden2mean.parameters():
        wght_norm = p.data.norm(2)
        grad_norm = p.grad.data.norm(2)
        h2m_wght_norm += wght_norm.item() ** 2
        h2m_grad_norm += grad_norm.item() ** 2
    # h2m_wght_norm = h2m_wght_norm ** (1. / 2)
    h2m_grad_norm = h2m_grad_norm ** (1. / 2)

    h2v_wght_norm = 0
    h2v_grad_norm = 0
    for p in model.hidden2logv.parameters():
        wght_norm = p.data.norm(2)
        grad_norm = p.grad.data.norm(2)
        h2v_wght_norm += wght_norm.item() ** 2
        h2v_grad_norm += grad_norm.item() ** 2
    # h2v_wght_norm = h2v_wght_norm ** (1. / 2)
    h2v_grad_norm = h2v_grad_norm ** (1. / 2)

    return (total_grad_norm, enc_grad_norm, dec_grad_norm, h2m_grad_norm, h2v_grad_norm)
