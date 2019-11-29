import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

# reproducibility
#torch.manual_seed(216)
#np.random.seed(216)

def to_cuda_var(x):
    """
    convert torch CUP to GPU
    :param x: torch tensor
    :return: torch tensor
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def drug_atc_path(drugs, atc_level = 4):
    """

    :param durgs: FDA drug lists
    :return: possible ATC path
    """
    df_atc = pd.read_csv('./data/fda_drugs/atc_fda_all.csv', index_col=0)
    df_atc_select = df_atc.loc[df_atc['ATC_LVL5'].isin(drugs)].copy()

    if atc_level == 1:
        df_atc_select.loc[:, 'ATC_PATH'] = df_atc_select.loc[:, 'ATC_LVL1'].values
    elif atc_level == 2:
        df_atc_select.loc[:, 'ATC_PATH'] = df_atc_select.loc[:, 'ATC_LVL1'].values + '||' + df_atc_select.loc[:, 'ATC_LVL2'].values
    elif atc_level == 3:
        df_atc_select.loc[:, 'ATC_PATH'] = df_atc_select.loc[:, 'ATC_LVL1'].values + '||' + df_atc_select.loc[:, 'ATC_LVL2'].values + '||' + df_atc_select.loc[:, 'ATC_LVL3'].values
    elif atc_level == 4:
        df_atc_select.loc[:, 'ATC_PATH'] = df_atc_select.loc[:, 'ATC_LVL1'].values + '||' + df_atc_select.loc[:, 'ATC_LVL2'].values + '||' + df_atc_select.loc[:, 'ATC_LVL3'].values + '||' + df_atc_select.loc[:, 'ATC_LVL4'].values
    return df_atc_select



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

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m, 1)
    return A[r, c]

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
        m2 = m**2
        m2 = np.where(m2<1.0, 1.0 + 1e-6, m2)
        dm = np.log(m + np.sqrt(m2 - 1))
        # prevent Inf. distance in dm
        dm = np.nan_to_num(dm)
        return upper_tri_indexing(dm) # convert to condense form
    elif manifold_type == 'Euclidean':
        dc = pdist(x, metric='euclidean')
        # prevent Inf. distance in dc
        dc = np.nan_to_num(dc)
        return dc

def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    X = np.concatenate((X), axis = 0)
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr