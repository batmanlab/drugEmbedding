#TODO: check potential numerical instability

import torch
from torch.distributions import MultivariateNormal
import numpy as np
from utils import to_cuda_var


def lorentz_to_poincare(h):
    if type(h) is torch.Tensor:
        return h[:,1:]/(h[:, 0]+1).view(-1, 1)
    elif type(h) is np.ndarray:
        return h[:, 1:] / (h[:, 0] + 1).reshape(-1, 1)


def poincare_dist(u, v):
    w = 1 + 2 * np.linalg.norm(u-v, ord=2, axis=1)/((1-np.linalg.norm(u, ord=2, axis=1))*(1-np.linalg.norm(v, ord=2, axis=1)))
    return np.arccosh(w)


def lorentz_model(z):
    [batch_size, n_h] = z.shape
    # check whether z are on the Lorentz manifold
    c1_idx = z[:,0] < 0
    lp = lorentz_product(z, z)
    c2_idx = torch.abs(lp+1) > 0.2
    c3_idx = torch.isnan(lp)
    c_idx = c1_idx.view(-1,1) | c2_idx | c3_idx
    # remove invalid z_samples
    if c_idx.sum().item()>0:
        mask_z = 1 - c_idx.repeat(1, n_h)
        c_idx = c_idx.unsqueeze_(-1)
        mask_cov = 1 - c_idx.repeat(1, n_h-1, n_h-1)
    else:
        mask_z = torch.ones(batch_size, n_h).byte()
        mask_cov = torch.ones(batch_size, n_h-1, n_h-1).byte()
    return mask_z, mask_cov


def lorentz_product(z1, z2):
    assert(z1.shape == z2.shape), 'Tensors have different dimensionality.'
    m = z1 * z2
    lor_prod = m[:, 1:].sum(dim=-1) - m[:, 0]
    lor_prod = torch.unsqueeze(lor_prod, dim=-1)
    return lor_prod


def lorentz_tangent_norm(x):
    lorentz_norm_x = lorentz_product(x, x)
    #lorentz_norm_x = torch.where(lorentz_norm_x < 0, torch.zeros_like(lorentz_norm_x)+1e-6, lorentz_norm_x) # make sure Loretnz norm > 0
    lorentz_norm_x = torch.clamp(lorentz_norm_x, min=1e-6)
    return torch.sqrt(lorentz_norm_x)


def parallel_transport(v, mu1, mu2):
    alpha = - lorentz_product(mu1, mu2)
    u = v + lorentz_product(mu2 - alpha*mu1, v)/(alpha+1)*(mu1+mu2)
    return u


def exp_map(u, mu_h):
    u_norm = lorentz_tangent_norm(u)
    z = torch.cosh(u_norm)*mu_h + torch.sinh(u_norm)*u/u_norm
    return z


def arccosh(x):
    x = torch.clamp(x, min=1.0 + 1e-6) # make sure x > 1
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def inv_exp_map(z,mu_h):
    alpha = -lorentz_product(mu_h, z)
    #alpha = torch.where(alpha <= 1, torch.ones_like(alpha)+1e-6, alpha) # make sure alpha > 1
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    u = arccosh(alpha)/torch.sqrt(torch.pow(alpha, 2)-1)*(z-alpha*mu_h)
    return u


def lorentz_mapping(x):
    # if the input is the origin of the Euclidean space
    [batch_size, n] = x.shape
    # interpret x_t as an element of tangent space of the origin of hyperbolic space
    x_t = torch.cat((to_cuda_var(torch.zeros(batch_size, 1)), x), 1)
    # origin of the hyperbolic space
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size, 1)), to_cuda_var(torch.zeros(batch_size, n))), 1)
    # exponential mapping
    z = exp_map(x_t, v0)
    return z


def lorentz_mapping_origin(x):
    batch_size, _ = x.shape
    return torch.cat((to_cuda_var(torch.ones(batch_size, 1)), x), 1)


def lorentz_sampling(mu_h, logvar):
    [batch_size, n_h] = mu_h.shape
    n = n_h - 1
    #step 1: Sample a vector (vt) from the Gaussian distribution N(0,COV) defined over R(n)
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu0)
    vt = mu0 + std * eps   # reparameterization trick
    #step 2: Interpret v as an element of tangent space of the origin of the hyperbolic space
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size,1)), to_cuda_var(torch.zeros(batch_size,n))),1)
    v = torch.cat((to_cuda_var(torch.zeros(batch_size, 1)), vt), 1)
    #step 3: Parallel transport the vector v to u which belongs to the tangent space of the mu
    u = parallel_transport(v, v0, mu_h)
    # step 4: Map u to hyperbolic space by exponential mapping
    z = exp_map(u, mu_h)
    return vt, u, z


def pseudo_hyperbolic_gaussian(z, mu_h, cov, version, vt=None, u=None):

    batch_size, n_h = mu_h.shape
    n = n_h - 1
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size, 1)), mu0), 1) # origin of the hyperbolic space

    # try not using inverse exp. mapping if vt is already known
    if vt is None and u is None:
        u = inv_exp_map(z, mu_h)
        v = parallel_transport(u, mu_h, v0)
        vt = v[:, 1:]
        logp_vt = (MultivariateNormal(mu0, cov).log_prob(vt)).view(-1, 1)
    else:
        logp_vt = (MultivariateNormal(mu0, cov).log_prob(vt)).view(-1, 1)

    r = lorentz_tangent_norm(u)

    if version == 1:
        alpha = -lorentz_product(v0, mu_h)
        log_det_proj_mu = n * (torch.log(torch.sinh(r)) - torch.log(r)) + torch.log(torch.cosh(r)) + torch.log(alpha)

    elif version == 2:
        log_det_proj_mu = (n-1) * (torch.log(torch.sinh(r))-torch.log(r))

    logp_z = logp_vt - log_det_proj_mu

    return logp_vt, logp_z
