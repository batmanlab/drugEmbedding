"""
synthetic branching diffusion

Tree object
1, One node is marked as Root node
2, Every node other than the root is associated with one parent node
3, Each node can have an arbiatry number of child node

"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

class Node:

    def __init__(self, x, J):
        self.J = J
        self.left = None
        self.right = None
        self.parent = None
        self.zx = None
        self.zy = None
        self.depth = None
        self.x = x
        self.y = self._get_noisy_obs()


    def _get_noisy_obs(self):
        y_lst = []
        for i in range(self.J):
            dim = len(self.x)
            mu = np.zeros(dim)
            cov = np.eye(dim) * 1/self.J
            y_lst.append((self.x + np.random.multivariate_normal(mu, cov, 1).reshape(dim,)).tolist())
        return np.asarray(y_lst)

class BranchingDiffusionDataset(Dataset):

    def __init__(self, nodes, J):
        self.depth = len(nodes)
        self.J = J
        self.nodes_dict, self.depth_dict = self._grow_tree(nodes)


    def _grow_tree(self, nodes):
        data_dict = {}
        depth_dict = {}
        nid = 1
        for depth in range(1, len(nodes)):
            for node in nodes[depth]:
                depth_dict[nid] = depth
                data_dict[nid] = node.x
                nid += 1
                for n in range(5):
                    depth_dict[nid] = depth
                    data_dict[nid] = node.y[n]
                    nid += 1
        return data_dict, depth_dict

    def __len__(self):
        return len(self.nodes_dict.keys())

    def __getitem__(self, idx):
        return {
        'idx': idx,
        'depth': self.depth_dict[idx+1],
        'data': self.nodes_dict[idx+1]}



"""
VAE Models
"""
from torch import nn, optim
from torch.nn import functional as F
from lorentz_model import *

class VAE(nn.Module):
    def __init__(self, dim_inputs, dim_h1, dim_z):
        super(VAE, self).__init__() # TODO: understand super() in class

        self.fc11 = nn.Linear(dim_inputs, dim_h1)

        self.fc_mu = nn.Linear(dim_h1, dim_z)
        self.fc_logvar = nn.Linear(dim_h1, 1)

        self.fc21 = nn.Linear(dim_z, dim_h1)
        self.fc22 = nn.Linear(dim_h1, dim_inputs)

    def encoder(self, x):
        h11 = F.relu(self.fc11(x))
        mu = self.fc_mu(h11)
        logvar = self.fc_logvar(h11)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        return z

    def decoder(self,z):
        h21 = F.relu(self.fc21(z))
        x_hat = self.fc22(h21)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, dim_inputs))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return mu, logvar, z, x_hat


class VAE_H(nn.Module):
    def __init__(self, dim_inputs, dim_h1, dim_z, version, model):
        super(VAE_H, self).__init__() # TODO: understand super() in class

        self.version = version #version of algorithm 2
        self.model = model # model = 'lorentz' or model = 'poincare'

        self.fc11 = nn.Linear(dim_inputs, dim_h1)

        self.fc_mu = nn.Linear(dim_h1, dim_z)
        self.fc_logvar = nn.Linear(dim_h1, 1)

        if model == 'lorentz':
            self.fc21 = nn.Linear(dim_z + 1, dim_h1)
        elif model == 'poincare':
            self.fc21 = nn.Linear(dim_z, dim_h1)
        self.fc22 = nn.Linear(dim_h1, dim_inputs)

    def encoder_h(self,x):
        h11 = F.relu(self.fc11(x))
        mu = self.fc_mu(h11)
        mu_h = lorentz_mapping(mu)
        logvar = self.fc_logvar(h11)
        return mu_h, logvar

    def reparameterize_h(self, mu_h, logvar):
        [batch_size, n_h] = mu_h.shape
        n = n_h - 1

        mu0 = torch.zeros(batch_size, n).float()
        mu0_h = lorentz_mapping_origin(mu0)
        diag = torch.eye(n).repeat(batch_size, 1, 1).float()

        #sampling z at mu_h on hyperbolic space
        cov = torch.exp(logvar).unsqueeze(dim=2) * diag
        vt, u, z = lorentz_sampling(mu_h, logvar)

        """
        # identify invalid z samples that are not on Lorentz manifold
        #mask_z, mask_cov = lorentz_model(z)

        # dropout bad z and corresponding parameters
        mu0 = torch.masked_select(mu0, mask_z[:,:n]).view(-1, n)
        mu0_h = torch.masked_select(mu0_h, mask_z).view(-1, n_h)
        mu_h = torch.masked_select(mu_h, mask_z).view(-1, n_h)
        cov = torch.masked_select(cov, mask_cov).view(-1, n, n)
        vt = torch.masked_select(vt, mask_z[:,:n]).view(-1, n)
        u = torch.masked_select(u, mask_z).view(-1, n_h)
        z = torch.masked_select(z, mask_z).view(-1, n_h)
        """

        """
        #caclulate log-pdf of posterior distribution q(z|x)
        logp_vt = (MultivariateNormal(mu0, cov).log_prob(vt)).view(-1, 1)
        r = lorentz_tangent_norm(u)
        if self.version == 1:
            alpha = -lorentz_product(mu0_h, mu_h)
            log_det_proj_mu = n * (torch.log(torch.sinh(r)) - torch.log(r)) + torch.log(torch.cosh(r)) + torch.log(alpha)
        elif self.version == 2:
            log_det_proj_mu = (n - 1) * (torch.log(torch.sinh(r)) - torch.log(r))
        logp_z = logp_vt - log_det_proj_mu
        """
        _, logp_z = pseudo_hyperbolic_gaussian(z, mu_h, cov, self.version)

        #calculate log-pdf of prior distribution p(mu0, I)
        _, logp_z0 = pseudo_hyperbolic_gaussian(z, mu0_h, diag, self.version)

        return z, logp_z0, logp_z

    def decoder_h(self, z):
        if self.model == 'poincare':
            z = lorentz_to_poincare(z)

        h21 = F.relu(self.fc21(z))
        x_hat = self.fc22(h21)
        return z, x_hat

    def forward(self, x):
        mu_h, logvar = self.encoder_h(x.view(-1, dim_inputs))
        z, logp_z0, logp_z = self.reparameterize_h(mu_h, logvar)
        z_p, x_hat = self.decoder_h(z)
        return mu_h, logvar, logp_z0, logp_z, x_hat, x, z_p


"""
loss functions
"""
def loss_fn(x, x_hat, mu, logvar, z, mc_flag):
    RECON = F.mse_loss(x_hat, x, reduction='sum')
    if mc_flag:
        KLD = kl_mc(z, mu, logvar)
    else:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON, KLD, RECON+KLD

def kl_mc(z, mu, logvar):
    [batch_size, dim_z] = z.shape
    diag = torch.eye(dim_z).repeat(batch_size,1,1)
    cov = torch.exp(logvar).unsqueeze(dim=2)*diag
    z_prior_pdf = MultivariateNormal(torch.zeros(dim_z), diag)
    logp_prior_z = z_prior_pdf.log_prob(z)
    z_posterior_pdf = MultivariateNormal(mu, cov)
    logp_posterior_z = z_posterior_pdf.log_prob(z)
    kl_loss = logp_posterior_z - logp_prior_z
    return torch.sum(kl_loss)

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    elif anneal_function == 'constant':
        return 1

def loss_fn_h(x, x_hat, logp_z0, logp_z):
    RECON = F.mse_loss(x_hat, x, reduction='sum')
    KLD = torch.sum(logp_z - logp_z0)
    return RECON, KLD, RECON+KLD

"""
generate dataset
"""
#initalize parameters in branching tree
dim_inputs = 50
depth = 6
J = 5
root = Node(np.zeros((dim_inputs)), J)
root.y = np.zeros((root.y.shape))
root.zx = np.zeros(2)
root.zy = np.zeros((J, 2))
root.depth = 0

#seed = 1657623761
#np.random.seed(seed)
#torch.manual_seed(seed)

nodes = [root]
p_nodes = [root]
for d in range(depth):
    num_p_nodes = len(p_nodes)
    c_nodes = []
    for n in range(num_p_nodes):
        p_node = p_nodes[n]
        l_node = Node(np.random.multivariate_normal(p_node.x,np.identity(dim_inputs),1).flatten(), J)
        l_node.parent = p_node
        c_nodes.append(l_node)

        r_node = Node(np.random.multivariate_normal(p_node.x,np.identity(dim_inputs),1).flatten(), J)
        r_node.parent = p_node
        c_nodes.append(r_node)
    p_nodes = c_nodes
    nodes.append(p_nodes)


#create branching diffusion dataset class
dataset = BranchingDiffusionDataset(nodes, J)

#initialize hyperparameters
batch_size = dataset.__len__()
dim_h1 = 200
dim_z = 2
log_interval = 10
num_epoch = 1000
epoch_interval = 10
learning_rate = 1e-3
mc_flag = False

version = 2
m1 = 'lorentz'
m2 = 'poincare'

annealing_function = 'constant'
#annealing_function = 'logistic'
k=0.0025
x0 = 2500


"""
Normal VAE
"""

method = 'VAE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if method == 'VAE':
    model = VAE(dim_inputs, dim_h1, dim_z).to(device)
elif method == 'VAE_H':
    model = VAE_H(dim_inputs, dim_h1, dim_z, version).to(device)

exp_dir = './experiments/BranchingDiffusion/vae_e_analytical'
summary_writer = SummaryWriter(log_dir=exp_dir)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

model.train()
model.float()
step = 0
for epoch in range(1, num_epoch+1):
    recon_loss_lst = []
    kl_loss_lst = []
    total_loss_lst = []
    for batch_idx, batch in enumerate(data_loader):
        data = batch['data'].float()
        data = data.to(device)
        optimizer.zero_grad()
        if method == 'VAE':
            mu, logvar, z, x_hat = model(data)
            recon_loss, kl_loss, loss = loss_fn(data, x_hat, mu, logvar, z, mc_flag)
        elif method == 'VAE_H':
            mu_h, logvar, logp_z0, logp_z, x_hat, x, z = model(data.float())
            recon_loss, kl_loss, _ = loss_fn_h(x_hat, x, logp_z0, logp_z)
            #KL annealing
            kl_weight = kl_anneal_function(annealing_function, step, k, x0)
            kl_loss = kl_weight * kl_loss
            loss = (recon_loss + kl_loss)

        loss.backward()
        optimizer.step()
        step += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon Loss: {:.6f}\tKL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data),
                recon_loss.item() / len(data),
                kl_loss.item() / len(data)))

        # keep tracking of loss terms
        recon_loss_lst.append(recon_loss.item()/len(data))
        kl_loss_lst.append(kl_loss.item()/len(data))
        total_loss_lst.append(loss.item()/len(data))

    #update tensorboardX
    summary_writer.add_scalar('Train Reconstruction loss', np.asarray(recon_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train KL loss', np.asarray(kl_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train Total loss', np.asarray(total_loss_lst).mean(), epoch)

#update Nodes class
z_np = z.detach().numpy()
nid = 0
for depth in range(1, len(nodes)):
    for n in range(np.power(2,depth)):
        nid = int(2*(1-np.power(2, depth-1))/(1-2) + n)
        node = nodes[depth][n]
        node.zx = z_np[nid*(J+1)]
        node.zy = z_np[nid*(J+1)+1:(nid+1)*(J+1)]

#plot x nodes
colors = ['b', 'g', 'r','c','m','y','k']
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
ax[0].scatter(nodes[0].zx[0], nodes[0].zx[1], marker = "x", c=colors[0], s=100)
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[0].plot(node.zx[0],node.zx[1], marker = "x", c= colors[depth])
        if node.parent is not None:
            ax[0].plot([node.zx[0], node.parent.zx[0]], [node.zx[1], node.parent.zx[1]], linestyle='-', color = colors[depth], linewidth = 7-depth)

#plot y nodes
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[0].scatter(node.zy[:,0],node.zy[:,1], marker = ".", c=colors[depth], s=depth, alpha=0.7)

title_loss = 'Recon Loss = ' + str(round(np.asarray(recon_loss_lst).mean(),1)) + ', KL Loss = ' + str(round(np.asarray(kl_loss_lst).mean(),1))
ax[0].set_title(title_loss)



"""
Hyperbolic VAE on the Lorentz Manifold
"""
method = 'VAE_H'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE_H(dim_inputs, dim_h1, dim_z, version, m1).to(device)

exp_dir = './experiments/BranchingDiffusion/vae_h_lorentz'
summary_writer = SummaryWriter(log_dir=exp_dir)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

model.train()
model.float()
step = 0
for epoch in range(1, num_epoch+1):
    recon_loss_lst = []
    kl_loss_lst = []
    total_loss_lst = []
    for batch_idx, batch in enumerate(data_loader):
        data = batch['data'].float()
        data = data.to(device)
        optimizer.zero_grad()
        if method == 'VAE':
            mu, logvar, z, x_hat = model(data)
            recon_loss, kl_loss, loss = loss_fn(data, x_hat, mu, logvar, z, mc_flag)
        elif method == 'VAE_H':
            mu_h, logvar, logp_z0, logp_z, x_hat, x, z = model(data.float())
            recon_loss, kl_loss, _ = loss_fn_h(x_hat, x, logp_z0, logp_z)
            # KL annealing
            kl_weight = kl_anneal_function(annealing_function, step, k, x0)
            loss = (recon_loss + kl_weight * kl_loss)

        loss.backward()
        optimizer.step()
        step += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon Loss: {:.6f}\tKL Loss: {:.6f}\tKL Weight: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data),
                recon_loss.item() / len(data),
                kl_loss.item() / len(data),
                kl_weight))

        # keep tracking of loss terms
        recon_loss_lst.append(recon_loss.item()/len(data))
        kl_loss_lst.append(kl_loss.item()/len(data))
        total_loss_lst.append(loss.item()/len(data))

    #update tensorboardX
    summary_writer.add_scalar('Train Reconstruction loss', np.asarray(recon_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train KL loss', np.asarray(kl_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train Total loss', np.asarray(total_loss_lst).mean(), epoch)

#update Nodes class
# transform to poincare disk
z = lorentz_to_poincare(z)
z_np = z.detach().numpy()
nid = 0
for depth in range(1, len(nodes)):
    for n in range(np.power(2,depth)):
        nid = int(2*(1-np.power(2, depth-1))/(1-2) + n)
        node = nodes[depth][n]
        node.zx = z_np[nid*(J+1)]
        node.zy = z_np[nid*(J+1)+1:(nid+1)*(J+1)]

#plot x nodes
ax[1].scatter(nodes[0].zx[0], nodes[0].zx[1], marker = "x", c=colors[0], s=100)
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[1].plot(node.zx[0],node.zx[1], marker = "x", c=colors[depth])
        if node.parent is not None:
            ax[1].plot([node.zx[0], node.parent.zx[0]], [node.zx[1], node.parent.zx[1]], linestyle='-', color=colors[depth], linewidth = 7-depth)

#plot y nodes
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[1].scatter(node.zy[:,0],node.zy[:,1], marker = ".", c=colors[depth], s=depth, alpha=0.7)

title_loss = 'Recon Loss = ' + str(round(np.asarray(recon_loss_lst).mean(),1)) + ', KL Loss = ' + str(round(np.asarray(kl_loss_lst).mean(),1))
ax[1].set_title(title_loss)

circle = plt.Circle((0,0),1,color='k',fill=False)
ax[1].add_artist(circle)
ax[1].set_xlim(-1.1,1.1)
ax[1].set_ylim(-1.1,1.1)



"""
Hyperbolic VAE on the Poincare disk
"""
method = 'VAE_H'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE_H(dim_inputs, dim_h1, dim_z, version, m2).to(device)

exp_dir = './experiments/BranchingDiffusion/vae_h_poincare'
summary_writer = SummaryWriter(log_dir=exp_dir)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

model.train()
model.float()
step = 0
for epoch in range(1, num_epoch+1):
    recon_loss_lst = []
    kl_loss_lst = []
    total_loss_lst = []
    for batch_idx, batch in enumerate(data_loader):
        data = batch['data'].float()
        data = data.to(device)
        optimizer.zero_grad()
        if method == 'VAE':
            mu, logvar, z, x_hat = model(data)
            recon_loss, kl_loss, loss = loss_fn(data, x_hat, mu, logvar, z, mc_flag)
        elif method == 'VAE_H':
            mu_h, logvar, logp_z0, logp_z, x_hat, x, z = model(data.float())
            recon_loss, kl_loss, _ = loss_fn_h(x_hat, x, logp_z0, logp_z)
            # KL annealing
            kl_weight = kl_anneal_function(annealing_function, step, k, x0)
            loss = (recon_loss + kl_weight * kl_loss)

        loss.backward()
        optimizer.step()
        step += 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRecon Loss: {:.6f}\tKL Loss: {:.6f}\tKL Weight: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data),
                recon_loss.item() / len(data),
                kl_loss.item() / len(data),
                kl_weight))

        # keep tracking of loss terms
        recon_loss_lst.append(recon_loss.item()/len(data))
        kl_loss_lst.append(kl_loss.item()/len(data))
        total_loss_lst.append(loss.item()/len(data))

    #update tensorboardX
    summary_writer.add_scalar('Train Reconstruction loss', np.asarray(recon_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train KL loss', np.asarray(kl_loss_lst).mean(), epoch)
    summary_writer.add_scalar('Train Total loss', np.asarray(total_loss_lst).mean(), epoch)

#update Nodes class
z_np = z.detach().numpy()
nid = 0
for depth in range(1, len(nodes)):
    for n in range(np.power(2,depth)):
        nid = int(2*(1-np.power(2, depth-1))/(1-2) + n)
        node = nodes[depth][n]
        node.zx = z_np[nid*(J+1)]
        node.zy = z_np[nid*(J+1)+1:(nid+1)*(J+1)]

#plot x nodes
ax[2].scatter(nodes[0].zx[0], nodes[0].zx[1], marker = "x", c=colors[0], s=100)
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[2].plot(node.zx[0],node.zx[1], marker = "x", c=colors[depth])
        if node.parent is not None:
            ax[2].plot([node.zx[0], node.parent.zx[0]], [node.zx[1], node.parent.zx[1]], linestyle='-', color=colors[depth], linewidth = 7-depth)

#plot y nodes
for depth in range(1, len(nodes)):
    for node in nodes[depth]:
        ax[2].scatter(node.zy[:,0],node.zy[:,1], marker = ".", c=colors[depth], s=depth, alpha=0.7)

title_loss = 'Recon Loss = ' + str(round(np.asarray(recon_loss_lst).mean(),1)) + ', KL Loss = ' + str(round(np.asarray(kl_loss_lst).mean(),1))
ax[2].set_title(title_loss)

circle = plt.Circle((0,0),1,color='k',fill=False)
ax[2].add_artist(circle)
ax[2].set_xlim(-1.1,1.1)
ax[2].set_ylim(-1.1,1.1)
fig.tight_layout()
plt.show()


stop = 0



























