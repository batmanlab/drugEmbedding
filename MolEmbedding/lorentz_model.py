import torch
from torch.distributions import MultivariateNormal
from utils import *

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
    """
    Debug: z can be really close to mu_h when latent dimension is low, e.g. n=2
           when this happens, alpha ~= 1, resulting the denominator of equation (6) 0
           so will be a Nan vector
    """
    assert(z1.shape == z2.shape),'Tensors have different dimensionality.'
    [batch_size, dim_z] = z1.shape
    lor_prod = -(z1[:,0]*z2[:,0]) + torch.bmm(z1[:,1:].view(batch_size, 1, dim_z-1), z2[:,1:].view(batch_size, dim_z-1, 1)).squeeze()
    return lor_prod.view(-1,1)

def lorentz_tangent_norm(x):
    """
    Debug: when x is very large, Lorentz Product may become negative, causing square root = Nan
           change negative value to a small positive number 1e-06
    """
    lorentz_norm_x = lorentz_product(x,x)
    lorentz_norm_x = torch.where(lorentz_norm_x<0, torch.zeros_like(lorentz_norm_x)+1e-6, lorentz_norm_x)
    return torch.sqrt(lorentz_norm_x)

def parallel_transport(v, mu1, mu2):
    alpha = - lorentz_product(mu1, mu2)
    u = v + lorentz_product(mu2 - alpha*mu1,v)/(alpha+1)*(mu1+mu2)
    return u

def exp_map(u,mu_h):
    u_norm = lorentz_tangent_norm(u)
    z = torch.cosh(u_norm)*mu_h + torch.sinh(u_norm)*u/u_norm
    """
    debug
    """
    if torch.isnan(z.sum()).item() == 1:
        stop = 0
    return z


def arccosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))

def inv_exp_map(z,mu_h):
    alpha = -lorentz_product(mu_h,z)
    """
    Numerical error:
    when z is too close to mu_h the lorentz inner product can become -0.99, then arcosh will beccome nan
    """
    alpha = torch.where(alpha<=1, torch.ones_like(alpha)+1e-6, alpha)
    u = arccosh(alpha)/torch.sqrt(torch.pow(alpha,2)-1)*(z-alpha*mu_h)
    """
    debug
    """
    if torch.isnan(u.sum()).item() == 1:
        stop = 0
    return u

def lorentz_mapping(x):
    # if the input is the origin of the Euclidean space
    [batch_size, n] = x.shape
    # interpret x_t as an element of tangent space of the origin of hyperbolic space
    x_t = torch.cat((to_cuda_var(torch.zeros(batch_size,1)),x),1)
    # origin of the hyperbolic space
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size,1)), to_cuda_var(torch.zeros(batch_size,n))),1)
    # exponential mapping
    z = exp_map(x_t, v0)
    return z

def lorentz_mapping_origin(x):
    [batch_size, n] = x.shape
    return torch.cat((to_cuda_var(torch.ones(batch_size,1)),x),1)


def lorentz_sampling(mu_h, logvar):
    [batch_size, n_h] = mu_h.shape
    n = n_h - 1
    #step 1: Sample a vector (vt) from the Gaussian distribution N(0,COV) defined over R(n)
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu0)
    vt = mu0 + std * eps
    #vt = MultivariateNormal(mu0, cov).sample()
    #step 2: Interpret v as an element of tangent space of the origin of the hyperbolic space
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size,1)), to_cuda_var(torch.zeros(batch_size,n))),1)
    v = torch.cat((to_cuda_var(torch.zeros(batch_size, 1)), vt), 1)
    #step 3: Parallel transport the vector v to u which belongs to the tangent space of the mu
    u = parallel_transport(v, v0, mu_h)
    # step 4: Map u to hyperbolic space by exponential mapping
    z = exp_map(u, mu_h)
    return vt, u, z

def pseudo_hyperbolic_gaussian(z, mu_h, cov, version, vt=None, u=None):
    [batch_size, n_h] = mu_h.shape
    n = n_h -1
    mu0 = to_cuda_var(torch.zeros(batch_size, n))
    v0 = torch.cat((to_cuda_var(torch.ones(batch_size, 1)), mu0), 1) #origin of the hyperbolic space

    """
    A Differentiable Gaussian-like Distribution on Hyperbolic Space for Gradient-Based Learning
    Nagano, 2019
    Algorithm 2
    """
    # try not using inverse exp. mapping if vt is already known
    if vt is None and u is None:
        u = inv_exp_map(z, mu_h)
        v = parallel_transport(u, mu_h, v0)
        vt = v[:,1:]

    logp_vt = (MultivariateNormal(mu0, cov).log_prob(vt)).view(-1, 1)
    r = lorentz_tangent_norm(u)

    if version ==1:

        """
        Algorithm V1
        """
        alpha = -lorentz_product(v0, mu_h)
        log_det_proj_mu = n * (torch.log(torch.sinh(r)) - torch.log(r)) + torch.log(torch.cosh(r)) + torch.log(alpha)

    elif version==2:
        """
        Algorithm V2
        """
        log_det_proj_mu = (n-1) * (torch.log(torch.sinh(r))-torch.log(r))

    logp_z = logp_vt - log_det_proj_mu

    """
    debug: z and mu_h are very close, causing r = 0
    """
    if torch.isnan(logp_z.sum()).item() == 1:
        stop = 0
    return logp_vt, logp_z

def lorentz_to_poincare(h):
    return h[:,1:]/(h[:,0]+1).view(-1,1)
