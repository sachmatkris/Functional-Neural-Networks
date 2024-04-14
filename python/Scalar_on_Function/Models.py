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
                  device=None, smoothed = True):
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
        self.func_covs = len(structure['func'])
        self.scalar_covs = structure['scalar'][1] - structure['scalar'][0]
        self.first_hidden = sub_hidden[0]

        self.smoothed = smoothed
        if smoothed == True:
            self.integration_length = 1000
            self.linspaces_integration = [np.linspace(start = 0, stop = 1, num = self.integration_length) for _ in range(self.func_covs)]
            self.func_bases = functional_bases
            self.func_values = [self.func_bases[i](self.linspaces_integration[i])[:,:,0] for i in range(self.func_covs)]
        elif smoothed == False:   
            self.linspaces_integration = [np.linspace(start = 0, stop = 1, num = structure['func'][k][1] - structure['func'][k][0]) for k in range(self.func_covs)]

        self.phi_values = [self.phi_bases[k](self.linspaces_integration[k])[:,:,0] for k in range(self.func_covs)]
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
            if self.smoothed == True:
                x_values = torch.from_numpy(FDataGrid(func_inputs[k], np.linspace(start = 0, stop = 1, num = func_inputs[k].shape[1])).to_basis(self.func_bases[k])(self.linspaces_integration[k])[:,:,0]).float().to(self.device)
            elif self.smoothed == False:
                x_values = func_inputs[k].float().to(self.device)
            phi_values = torch.from_numpy(self.phi_values[k]).float().to(self.device)
            integral = torch.trapezoid(torch.einsum('ij,mj->imj', x_values, phi_values), torch.from_numpy(self.linspaces_integration[k]).float().to(self.device), dim = 2) # 2nd parameter - dx
            summand_funcs_sep[:,k,:] = torch.einsum('hm,im->ih', self.functional_params[k], integral)
        summand_func = torch.sum(summand_funcs_sep, dim = 1)
        summand_scal = torch.einsum('hj,ij->ih', self.scalar_params, scalar_inputs)
        hidden_nodes = summand_func + summand_scal + self.bias
        return(self.FF(F.relu(self.layer_norm(hidden_nodes))))

    def beta_weight(self, fnn, start_end):
        # start_end : list of tuples where each tuple indicates (start, end) of each functional input
        beta = []
        for k in range(fnn.func_covs):
            functional_params = fnn.functional_params[k].cpu().detach()
            phi_values = torch.from_numpy(fnn.phi_values[k]).float()
            beta.append(torch.einsum('km,mt->kt', functional_params, phi_values).mean(dim = 0))
        return [{'x' : np.linspace(start_end[k][0], start_end[k][1], fnn.integration_length), 'y' : beta[k]} for k in range(fnn.func_covs)]
    



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

    

   

class NN(nn.Module):
    def __init__(self, in_d, sub_hidden, dropout, num_classes = 1, last_layer = nn.Identity(), model_version = "advanced", device = None):
        super().__init__() 
        self.device = device
        self.FF = FeedForward(in_d, sub_hidden, num_classes = num_classes, model_version = model_version, last_layer = last_layer, dropout = dropout).to(device)
        
    def forward(self, x):
        return self.FF(x.to(self.device))
    


class ConvLayers(nn.Module):
    def __init__(self,  in_d = 100, hidden_channels = [4,4], kernel_convolution = 8, kernel_pool = 4, convolution_stride = 1, pool_stride = 2, dropout=0.1):
        """
        in_d                 : dimensions of the functional covariate
        hidden_channels      : list of convolutional layer channels
        kernel_convolution   : length of kernel for conv1d
        kernel_pool          : length of kernel for pool
        convolution_stride   : length of stride for conv1d
        pool_stride          : length of stride for pool
        dropout              : dropout probability, a float between 0.0 and 1.0
        """
        super().__init__()
        dim = [1] + hidden_channels
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels = dim[i-1], out_channels = dim[i], kernel_size = kernel_convolution, stride = convolution_stride) for i in range(1, len(dim))])
        self.pool = nn.MaxPool1d(kernel_size = kernel_pool, stride = pool_stride).to(device)
        self.dp = nn.Dropout(dropout)
        
        def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
            return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        L_in = in_d
        for _ in range(1, len(dim)):
            L_out = calculate_output_length(L_in, kernel_convolution, convolution_stride)
            L_out = calculate_output_length(L_out, kernel_pool, pool_stride)
            L_in = L_out
        self.out_d = L_in * hidden_channels[-1]
    
    def forward(self, t):
        for i in range(len(self.conv_layers)):
            t = self.pool(F.relu(self.conv_layers[i](t)))
            # apply dropout
            t = self.dp(t)
        return t



class CNN(nn.Module):
    def __init__(self, structure, conv_hidden_channels = [4,4], fc_hidden = [24, 24],
                 kernel_convolution = 8, kernel_pool = 4, convolution_stride = 1, pool_stride = 2,
                 num_classes = 1, dropout = 0.1, last_layer = nn.Identity(), device = None):  
        """     
        structure            : dict with keys 'func' and 'scalar' defining the structure of functional and scalar parts
        conv_hidden_channels : list of channels in each conv1d layer
        fc_hidden            : list of in_d for each hidden fully connected layer
        kernel_convolution   : length of kernel for conv1d
        kernel_pool          : length of kernel for pool
        convolution_stride   : length of stride for conv1d
        pool_stride          : length of stride for pool
        dropout              : dropout probability, a float between 0.0 and 1.0
        activation           : activation function at each layer
        """
        super().__init__()
        self.structure = structure
        self.func_in_d = [func_cov[1] - func_cov[0] for func_cov in structure['func']]
        self.scalar_in_d = structure['scalar'][1] - structure['scalar'][0]
        self.device = device
        self.conv_layers = nn.ModuleList([ConvLayers(in_d, conv_hidden_channels, kernel_convolution, kernel_pool, convolution_stride, pool_stride, dropout)
                                           for in_d in self.func_in_d]).to(device)
        self.out_d_values = [layer.out_d for layer in self.conv_layers]
        fc_in_d = sum(self.out_d_values) + self.scalar_in_d
        self.fc_layers = FeedForward(in_d = fc_in_d, hidden = fc_hidden, num_classes = num_classes, dropout = dropout, last_layer = last_layer, model_version = "simple").to(device)
    
    def forward(self, x):
        func_inputs = [x[:, func_cov[0] : func_cov[1]].unsqueeze(1).to(self.device) for func_cov in self.structure['func']]
        scalar_inputs = x[:, self.structure['scalar'][0] : self.structure['scalar'][1]].reshape(-1, self.scalar_in_d).to(self.device) if self.scalar_in_d != 0 else torch.tensor([]).to(self.device)
        result = []
        for layer, input_data in zip(self.conv_layers, func_inputs):
            output = layer(input_data)
            result.append(output)
        conv_output = torch.hstack([torch.flatten(torch.hstack(result), 1), scalar_inputs])
        return(self.fc_layers(conv_output))



class LSTM(nn.Module):
    def __init__(self, structure, lstm_hidden = [50], fc_hidden = [24, 24], num_layers = 1, bidirectional = True, num_classes = 1, dropout = 0, last_layer = nn.Identity(), device = None):
        """
        structure            : dict with keys 'func' and 'scalar' defining the structure of functional and scalar parts
        lstm_hidden          : list of hidden sizes in LSTM for each functional covariate
        fc_hidden            : list of in_d for each hidden fully connected layer
        num_layers           : number of layers in each LSTM
        bidirectional        : True/False
        dropout              : dropout probability, a float between 0.0 and 1.0
        activation           : activation function at each layer        
        """
        super().__init__()
        self.structure = structure
        self.num_layers = num_layers
        self.D = 2 if bidirectional == True else 1
        self.func_in_d = [func_cov[1] - func_cov[0] for func_cov in structure['func']]
        self.scalar_in_d = structure['scalar'][1] - structure['scalar'][0]
        self.lstm_hidden = lstm_hidden
        #torch.manual_seed(3)
        self.lstm = nn.ModuleList([nn.LSTM(input_size = 1, hidden_size = lstm_hidden[i], num_layers = num_layers,
                                           bidirectional = bidirectional, batch_first = True).to(device) for i in range(len(self.func_in_d))])
        fc_in_d = sum(lstm_hidden) * self.D + self.scalar_in_d
        self.fc_layers = FeedForward(in_d = fc_in_d, hidden = fc_hidden, num_classes = num_classes, dropout = dropout, last_layer = last_layer, model_version = "simple").to(device)
        self.device = device

    def forward(self, x):
        func_inputs = [x[:, func_cov[0] : func_cov[1]].unsqueeze(2).to(self.device) for func_cov in self.structure['func']]
        scalar_inputs = x[:, self.structure['scalar'][0] : self.structure['scalar'][1]].reshape(-1, self.scalar_in_d).to(self.device) if self.scalar_in_d != 0 else torch.tensor([]).to(self.device)
        lstm_out = []
        hidden = [torch.randn(self.D * self.num_layers, len(x), dim).to(self.device) for dim in self.lstm_hidden]
        carry = [torch.randn(self.D * self.num_layers, len(x), dim).to(self.device) for dim in self.lstm_hidden]
        for i  in range(len(self.lstm)):
            lstm_out.append(self.lstm[i](func_inputs[i], (hidden[i], carry[i]))[0][:,-1])
        t = torch.hstack(lstm_out + [scalar_inputs])
        return self.fc_layers(t)
