import matplotlib.pyplot as plt
from matplotlib import gridspec

time = [0, 1, 2, 3, 4]
y1 = range(10, 15)
y2 = range(15, 20)

# Create a figure and gridspec
fig = plt.figure(figsize=(5, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

# Create subplots
axs = [fig.add_subplot(gs[i]) for i in range(2)]

# Plot data on each subplot
axs[0].plot(time, y1)
axs[1].plot(time, y2)

plt.show()

def weight_plots(fnn, adafnn, num_func = 1):
    fig = plt.figure(figsize = (16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    axs = [fig.add_subplot(gs[i]) for i in range(2)]
    axs[0] = fnn.beta_weight()













import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
import pandas as pd

from skfda import FDataGrid
from skfda.representation.basis import  FourierBasis, BSplineBasis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils


task = 1
beta, g, snr = 1, 1, 0.5
data_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/task {task}/B{beta}_G{g}/snr{snr}/'
save_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/task {task}/B{beta}_G{g}/snr{snr}/'
Y_dir = f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'

X = pd.read_csv(data_directory + 'X/X.csv', header = None).values
T = pd.read_csv(data_directory + 'T/T.csv', header = None).values
Y = torch.from_numpy(pd.read_csv(data_directory + Y_dir, header = None).values).float()
cv_folds = Utils.kfold_cv(X)
structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
loss = nn.MSELoss()

EPOCHS = 300
train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, 0, 'FNN', batch_size = 16)
train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, 0, 'AdaFNN', batch_size = 16)

# FNN
model_FNN = Models.FNN(structure = structure, functional_bases = [BSplineBasis(n_basis = 5)],
                    phi_bases = [FourierBasis(n_basis = 5)], sub_hidden = [32, 32],
                    dropout = 0, device = device)
fnn, fnn_acc = Utils.pytorch_trainer_model(model_FNN, 'FNN', loss, 'regression', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.001, device = 'cuda:0')

# AdaFNN
model_AdaFNN = Models.AdaFNN(structure = structure, n_bases = [7], bases_hidden = [[64, 64, 64]], sub_hidden = [16, 16, 16],
                    lambda1 = 0.1, lambda2 = 0.9, dropout = 0, device = device)     
adafnn, adafnn_acc = Utils.pytorch_trainer_model(model_AdaFNN, 'AdaFNN', loss, 'regression', train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.01, device = 'cuda:0')











def plot_bases(adafnn, device):
    bases_all = []    
    with torch.inference_mode():
        for k in range(adafnn.func_covs): 
            bases_individual = []    
            for i, basis in enumerate(adafnn.BL[k]):
                t = adafnn.grids[k]
                T = torch.from_numpy(t).to(device).float().unsqueeze(-1)
                y = np.squeeze(basis(T).squeeze(dim=-1).detach().cpu().numpy())
                y_sq = y ** 2
                l2_norm = np.sqrt(np.sum((y_sq[:-1] + y_sq[1:]) * (t[1:] - t[:-1])) / 2)
                bases_individual.append(y / l2_norm) 
            #bases_all.append(sum(bases_individual))
            bases_all.append(bases_individual)
        #B = len(bases_all)
        B = len(bases_individual)
        fig, axs = plt.subplots(1, B, squeeze = False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        for i in range(B):        
            axs[0][i].plot(t, bases_individual[i], linewidth=3.5, label="basis"+str(i+1))
            axs[0][i].legend()
        return bases_individual
plot_bases(adafnn, device)
plt.show()






def weight_plots(fnn, adafnn, num_func = 1):
    fig, axs = plt.subplots(2, num_func, figsize = (18, 9), squeeze=False)
    beta_fnn = fnn.beta_weight(fnn, [(-2, 4)])
    [axs[0][k].plot(beta_fnn[k]['x'], beta_fnn[k]['y']) for k in range(num_func)]
    [axs[1][k].plot(beta_fnn[k]['x'], beta_fnn[k]['y']) for k in range(num_func)]
    plt.show()

weight_plots(fnn, adafnn)

















import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import  FourierBasis, BSplineBasis
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta
    


class FeedForward(nn.Module):
    def __init__(self, in_d=1, hidden=[4,4,4], num_classes = 1,  dropout=0.1, activation = F.relu, last_layer = nn.Identity(), model_version = 'advanced'):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # num_classes : number of classes, if regression - 1 (default), for classification, set to K classes
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.activation = activation
        self.last_layer = last_layer
        self.model_version = model_version
        dim = [in_d] + hidden + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        if model_version == 'simple':
            pass
        elif model_version == 'advanced':
            self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
            self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])
        else:
            print('Please correctly specify model version: "simple" or "advanced"')

    def forward(self, t):
        if self.model_version == 'simple':     
            for i in range(len(self.layers)-1):
                t = self.layers[i](t)
                t = self.activation(t)
            # linear activation at the last layer
            return self.last_layer(self.layers[-1](t))
        elif self.model_version == 'advanced':
            for i in range(len(self.layers)-1):
                t = self.layers[i](t)
                # skipping connection
                t = t + self.ln[i](t)
                t = self.activation(t)
                # apply dropout
                t = self.dp[i](t)
            # linear activation at the last layer
            return self.last_layer(self.layers[-1](t))



class FNN(nn.Module):
    def __init__(self, structure, functional_bases=[FourierBasis([0,1], 5)], phi_bases=[FourierBasis([0,1], 3)], 
                  sub_hidden=[128, 128, 128], num_classes = 1, dropout=0, last_layer = nn.Identity(),
                  device=None):
        """
        structure        : dict with keys 'func' and 'scalar' defining the structure of functional and scalar parts
        functional_bases : list of basis objects for input covariates
        phi_bases        : list of basis objects for weights 
        sub_hidden       : hidden layers in the subsequent network, array of integers
        grid             : list of observation time grid, array of sorted floats including 0.0 and 1.0
        num_classes      : number of classes, if regression - 1 (default), for classification, set to K classes
        dropout          : dropout probability
        device           : device for the training
        """
        super().__init__() 
        # no reason to pass basis objects, better alternative to pass basis coefficients for each func_cov (FUNC DATA)
        self.phi_bases = phi_bases
        self.beta_bases_total = sum([basis.n_basis for basis in self.phi_bases])
        self.func_bases = functional_bases
        self.func_covs = len(structure['func'])
        self.scalar_covs = structure['scalar'][1] - structure['scalar'][0]
        integration_length = 1000
        self.linspaces_integration = [np.linspace(start = 0, stop = 1, num = integration_length) for _ in range(self.func_covs)]
        self.first_hidden = sub_hidden[0]

        self.phi_values = [self.phi_bases[i](self.linspaces_integration[i])[:,:,0] for i in range(self.func_covs)]
        self.func_values = [self.func_bases[i](self.linspaces_integration[i])[:,:,0] for i in range(self.func_covs)]

        self.structure = structure
        self.device = device

        bound = 1/math.sqrt(self.first_hidden)
        self.functional_params = nn.ParameterList([(-bound - bound) * torch.rand([self.first_hidden, basis.n_basis], requires_grad=True).float() + bound for basis in phi_bases]).to(device) # kim
        self.scalar_params = nn.Parameter((-bound - bound) * torch.rand([self.first_hidden, self.scalar_covs], requires_grad=True).float() + bound).to(device)
        self.bias = nn.Parameter((-bound - bound) * torch.rand([self.first_hidden], requires_grad=True).float() + bound).to(device)
        self.FF = FeedForward(in_d = sub_hidden[0], hidden = sub_hidden[1:], num_classes = num_classes, last_layer = last_layer, dropout = dropout).to(device)
        self.layer_norm = nn.LayerNorm(self.first_hidden).to(device)


    def forward(self, x):
        func_inputs = [x[:, self.structure['func'][i][0] : self.structure['func'][i][1]] for i in range(self.func_covs)]
        scalar_inputs = x[:, self.structure['scalar'][0] : self.structure['scalar'][1]].to(self.device)
      
        summand_funcs_sep = torch.empty([x.shape[0], self.func_covs, self.first_hidden]).to(self.device)
        for k in range(self.func_covs):
            x_values = torch.from_numpy(FDataGrid(func_inputs[k], np.linspace(start = 0, stop = 1, num = func_inputs[k].shape[1])).to_basis(self.func_bases[k])(self.linspaces_integration[k])[:,:,0]).float().to(self.device)
            phi_values = torch.from_numpy(self.phi_values[k]).float().to(self.device)
            integral = torch.trapezoid(torch.einsum('ij,mj->imj', x_values, phi_values), torch.from_numpy(self.linspaces_integration[k]).float().to(self.device), dim = 2) # 2nd parameter - dx
            summand_funcs_sep[:,k,:] = torch.einsum('hm,im->ih', self.functional_params[k], integral)
        summand_func = torch.sum(summand_funcs_sep, dim = 1)
        summand_scal = torch.einsum('hj,ij->ih', self.scalar_params, scalar_inputs)
        hidden_nodes = summand_func + summand_scal + self.bias
        return(self.FF(F.relu(self.layer_norm(hidden_nodes))))

    def beta_weight(self):
        beta = []
        fig, ax = plt.subplots(self.func_covs, 1, figsize = (16, 9))
        for k in range(self.func_covs):
            functional_params = self.functional_params[k].cpu().detach()
            phi_values = torch.from_numpy(self.phi_values[k]).float()
            beta.append(torch.einsum('km,mt->t', functional_params, phi_values))
            ax[k].plot(self.linspaces_integration[k], beta[k])
        return fig, ax



class AdaFNN(nn.Module):
    def __init__(self, structure, n_bases = [4],
                 bases_hidden = [[64, 64]], sub_hidden =  [24, 24], 
                 lambda1 = 0.0, lambda2 = 0.0, num_classes = 1, dropout = 0,
                 last_layer = nn.Identity(), device = None):
        """
        structure         : dict with keys 'func' and 'scalar' defining the structure of functional and scalar parts
        n_bases           : list of number of basis nodes (per func cov)
        bases_hidden      : list of lists of hidden layers used in each basis node
        sub_hidden        : hidden layers in the subsequent network, array of integers
        lambda1           : penalty of L1 regularization, a positive real number
        lambda2           : penalty of L2 regularization, a positive real number
        num_classes       : number of classes, if regression - 1 (default), for classification, set to K classes
        dropout           : dropout probability
        device            : device for the training
        """
        super().__init__()

        self.structure = structure
        self.func_covs = len(structure['func'])
        self.scalar_covs = structure['scalar'][1] - structure['scalar'][0]
        self.bases_total = sum(n_bases)

        func_in_d = [func_cov[1] - func_cov[0] for func_cov in structure['func']]
        self.grids = [np.linspace(start = 0, stop = 1, num = integration_grid) for integration_grid in func_in_d]
        self.n_bases = n_bases
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        # grid should include both end points
        # send the time grid tensor to device
        self.t = [torch.from_numpy(grid).float().to(device) for grid in self.grids]
        self.h = [torch.from_numpy(grid[1:] - grid[:-1]).float().to(device) for grid in self.grids]
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([nn.ModuleList([FeedForward(hidden = bases_hidden[i], dropout = dropout, model_version = 'advanced').to(device)
                                                 for _ in range(n_bases[i])])
                                  for i in range(self.func_covs)])
        # instantiate the subsequent network, we add scalar covs + no of functional integrals (sum(n_bases[i]))    
        self.FF = FeedForward(in_d = sum(n_bases) + self.scalar_covs, hidden = sub_hidden, num_classes = num_classes, last_layer = last_layer, dropout = dropout).to(device)
    
    
    def _inner_product(self, f1, f2, t):
        #prod_old = f1 * f2 # (B, J = len(h) + 1)
        #prod_old = torch.matmul((prod_old[:, :-1] + prod_old[:, 1:]), h.unsqueeze(dim=-1))/2
        prod = torch.trapezoid(f1 * f2, t.to(self.device), dim = 1).unsqueeze(1) # (B, J = len(h) + 1)
        return prod

    def _l1(self, f, h):
        # f dimension : ( B bases, J )
        B, J = f.size()
        return self._inner_product(torch.abs(f).to(self.device), torch.ones((B, J)).to(self.device), h)

    def _l2(self, f, h):
        # f dimension : ( B bases, J )
        # output dimension - ( B bases, 1 )
        return torch.sqrt(self._inner_product(f, f, h)) 
    
    def forward(self, x):           
        func_inputs = [x[:, func_cov[0] : func_cov[1]].to(self.device) for func_cov in self.structure['func']]
        scalar_inputs = x[:, self.structure['scalar'][0] : self.structure['scalar'][1]].to(self.device)
        self.bases = []
        for i in range(self.func_covs):
            grid = self.t[i]
            T = grid.unsqueeze(-1)
            self.bases.append([basis(T).transpose(-1, -2) for basis in self.BL[i]])
        l2_norms = [self._l2(torch.cat(self.bases[i], dim=0), self.t[i]).detach() for i in range(self.func_covs)]
        self.normalized_bases = [[self.bases[i][j] / (l2_norms[i][j, 0] + 1e-6) for j in range(self.n_bases[i])] for i in range(self.func_covs)]
        score = torch.cat([self._inner_product(b.repeat((x.shape[0], 1)), func_inputs[i], self.t[i]) for i in range(self.func_covs)
                            for b in self.bases[i]], dim=-1)
        final_inputs = torch.cat([score, scalar_inputs], dim = 1)
        return self.FF(final_inputs)        


    def R1(self, l1_k = 0):
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        # sample l1_k basis nodes to regularize
        selected = [np.random.choice(base, min(l1_k, base), replace=False) for base in self.n_bases] if l1_k != 0 else [np.arange(base) for base in self.n_bases]
        selected_bases = [torch.cat([self.normalized_bases[k][i] for i in selected[k]], dim=0).to(device) for k in range(self.func_covs)]# (k, J)       
        return torch.sum(torch.Tensor([self.lambda1 * torch.mean(self._l1(selected_bases[k], self.t[k])) for k in range(self.func_covs)]))


    def R2(self, l2_pairs = 0):
        if self.lambda2 == 0 or self.n_bases == 1: return torch.zeros(1).to(self.device)
        l2_k = [min(l2_pairs, base * (base - 1) // 2) for base in self.n_bases] if l2_pairs != 0 else [base * (base - 1) // 2 for base in self.n_bases]
        f1, f2 = [[None] * k for k in l2_k], [[None] * k for k in l2_k]
        R2_func_covs = [None] * self.func_covs
        for k in range(self.func_covs):
            for i in range(l2_k[k]):
                a, b = np.random.choice(self.n_bases[k], 2, replace=False)
                f1[k][i], f2[k][i] = self.normalized_bases[k][a], self.normalized_bases[k][b]
            R2_func_covs[k] = self.lambda2 * torch.mean(torch.abs(self._inner_product(torch.cat(f1[k], dim=0).to(device),
                                                                  torch.cat(f2[k], dim=0).to(device),
                                                                  self.t[k])))
        return torch.sum(torch.Tensor(R2_func_covs))













