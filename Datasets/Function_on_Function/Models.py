import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import  FourierBasis, BSplineBasis
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class FFBNN(nn.Module):
    def __init__(self, bases = {'input':[FourierBasis(n_basis = 25)], 'hidden':FourierBasis(n_basis = 17)},
                  inc_nodes=1, hidden_nodes=[8,8,8], out_nodes=1, q = 24,
                  lambda_weight = 0.5, lambda_bias = 0.5, device = None):
        """
        base         : skfda basis object
        inc_nodes    : number of functional covariates
        hidden_nodes : list with hidden nodes in each layer (length of the list = # of layers)
        out_nodes    : number of functional responses
        q            : grid over which the functions are evaluated throughout the layers
        """
        super().__init__() 
        self.inc_nodes = inc_nodes
        self.hidden_nodes = hidden_nodes
        self.out_nodes = out_nodes
        self.q = q
        self.lambda_weight = lambda_weight
        self.lambda_bias = lambda_bias
        self.device = device 
        self.bases = bases
        self.n_basis = {'input' : [bases['input'][i].n_basis for i in range(inc_nodes)], 'hidden' : bases['hidden'].n_basis}
        
        T = np.linspace(0, 1, q)
        self.v_t_1 = [torch.from_numpy(bases['input'][i](T).squeeze(2)).float().to(device) for i in range(inc_nodes)]
        self.v_s = torch.from_numpy(bases['hidden'](T).squeeze(2)).float().to(device)
        self.v_t_1_d2 = [torch.from_numpy(bases['input'][i].derivative(order = 2)(T).squeeze(2)).float().to(device) for i in range(inc_nodes)]
        self.v_s_d2 = torch.from_numpy(bases['hidden'].derivative(order = 2)(T).squeeze(2)).float().to(device)
        self.dx = 1/(q-1)

        node_structure = [inc_nodes] + hidden_nodes + [out_nodes]

        def layer_maker(inc_nodes, out_nodes, n_basis):
            W_l = nn.Parameter(torch.randn([inc_nodes, out_nodes, n_basis, n_basis], requires_grad=True).float())
            B_l = nn.Parameter(torch.zeros([out_nodes, n_basis], requires_grad=True).float())
            return nn.ParameterList((W_l,B_l))
        
        self.layers = [nn.ParameterList(
                            (nn.ParameterList([torch.randn([node_structure[1], self.n_basis['input'][i], self.n_basis['hidden']]) for i in range(inc_nodes)]),
                            nn.Parameter(torch.zeros([node_structure[1], self.n_basis['hidden']], requires_grad=True).float())))]      
        for i in range(1, len(node_structure) - 1):
            self.layers.append(layer_maker(node_structure[i], node_structure[i+1], self.n_basis['hidden']))          
        self.layers = nn.ModuleList(self.layers).to(device)
        self.relu = nn.ReLU()




    def forward(self, x):
        
        x = x.to(self.device)
        def layer_calc(x, theta, dx = self.dx):
            W_l, B_l = theta
            A_l_integrand = torch.einsum('pq,ijq->ijpq ', self.v_s, x)    
            A_l = torch.trapezoid(A_l_integrand, dx = dx, dim = 3)
            vA = torch.einsum('bq,ijp->ijbpq', self.v_s, A_l)
            WvA = torch.einsum('jkpb,ijbpq->ikq', W_l, vA)
            bias = torch.einsum('kb,bq->kq', B_l, self.v_s)
            H_l = bias + WvA
            return(H_l)
               
        W_0, B_0 = self.layers[0] 
        out = []
        for k in range(self.inc_nodes):
            A_l_integrand = torch.einsum('pq,iq->ipq ', self.v_t_1[k], x[:,k,:])
            A_l = torch.trapezoid(A_l_integrand, dx = self.dx, dim = 2)
            vA = torch.einsum('bq,ip->ibpq', self.v_s, A_l)
            WvA = torch.einsum('kpb,ibpq->ikq', W_0[k], vA) ### MAYBE????
            out.append(WvA)
        sum_WvA = torch.sum(torch.stack(out, dim = 1), dim = 1)
        bias = torch.einsum('kb,bq->kq', B_0, self.v_s)
        H_1 = bias + sum_WvA  
        x = self.relu(H_1)
        for i in range(1, len(self.layers) - 1):
            x = self.relu(layer_calc(x.to(device), self.layers[i]))        
        return layer_calc(x, self.layers[-1])
    
     

    def regularization(self):
        
        def weight_grad(self, W):
            out_s1_d2 = torch.einsum('xp,jkpb,bz->jkxz', self.v_s_d2.T, W, self.v_s)
            out_s2_d2 = torch.einsum('xp,jkpb,bz->jkxz', self.v_s.T, W, self.v_s_d2)
            out_d2 = (out_s1_d2 + out_s2_d2)**2
            out_int = torch.trapezoid(torch.trapezoid(out_d2, dx = self.dx, dim = 3), dx = self.dx, dim = 2)
            return torch.mean(out_int, dim = [0, 1])
        
        def bias_grad(self, B):
            b_d2 = (B @ self.v_s_d2)**2
            b_int = torch.trapezoid(b_d2, dx = self.dx, dim = 1)
            return torch.mean(b_int)

        weight = 0
        for j in range(self.inc_nodes):
            out_s_d2 = torch.einsum('xp,kpb,bz->kxz', self.v_t_1[j].T, self.layers[0][0][j], self.v_s_d2)
            out_t_d2 = torch.einsum('xp,kpb,bz->kxz', self.v_t_1_d2[j].T, self.layers[0][0][j], self.v_s)
            out_d2 = (out_s_d2 + out_t_d2)**2
            out_int = torch.trapezoid(torch.trapezoid(out_d2, dx = self.dx, dim = 2), dx = self.dx, dim = 1)
            weight += torch.sum(out_int)
        weight /= self.inc_nodes
        bias = bias_grad(self, B = self.layers[0][1])

        for l in range(1, len(self.layers)):
            bias += bias_grad(self, B = self.layers[l][1])
            weight += weight_grad(self, W = self.layers[l][0])
        
        return self.lambda_weight/len(self.layers) * weight + self.lambda_bias/len(self.layers) * bias





class FFDNN(nn.Module):
    def __init__(self, inc_nodes = 1, hidden_nodes = [8,8,8], out_nodes = 1,
                  q = {'in' : [24], 'hidden' : 10, 'out' : 24},
                  lambda_weight = 0.5, lambda_bias = 0.5, device = None):
        """
        inc_nodes    : number of functional covariates
        hidden_nodes : list with hidden nodes in each layer (length of the list = # of layers)
        out_nodes    : number of functional responses
        q            : a dictionary with keys 'in' (list), 'hidden' (scalar), 'out' (scalar)
        """
        super().__init__() 
        self.inc_nodes = inc_nodes
        self.hidden_nodes = hidden_nodes
        self.out_nodes = out_nodes
        self.q = q
        self.lambda_weight = lambda_weight
        self.lambda_bias = lambda_bias
        self.device = device 
        
        self.dx = {'in' : [1/(q_ - 1) for q_ in q['in']], 'hidden' : 1/(q['hidden'] - 1), 'out' : 1/(q['out'] - 1)}
        node_structure = [inc_nodes] + hidden_nodes + [out_nodes]

        def layer_maker(inc_nodes, out_nodes, q_in, q_out):
            W_l = nn.Parameter(torch.randn([inc_nodes, out_nodes, q_in, q_out], requires_grad=True).float())
            B_l = nn.Parameter(torch.zeros([out_nodes, q_out], requires_grad=True).float())
            return nn.ParameterList((W_l,B_l))
        
        self.layers = [nn.ParameterList(
                            (nn.ParameterList([torch.randn([node_structure[1], q['in'][i], q['hidden']]) for i in range(inc_nodes)]),
                            nn.Parameter(torch.zeros([node_structure[1], q['hidden']], requires_grad=True).float())))]      
        for i in range(1, len(node_structure) - 2):
            self.layers.append(layer_maker(node_structure[i], node_structure[i+1], q['hidden'], q['hidden']))
        self.layers.append(layer_maker(node_structure[-2], node_structure[-1], q['hidden'], q['out']))         
        self.layers = nn.ModuleList(self.layers).to(device)
        self.relu = nn.ReLU()


    def forward(self, x):     
        
        x = x.to(self.device)
        def layer_calc(x, theta, dx):
            W_l, B_l = theta
            integrand = torch.einsum('jkpq,ijp->ijkpq', W_l, x) 
            integral = torch.trapezoid(integrand, dx = dx, dim = 3)
            summation = torch.sum(integral, 1)
            H_l = B_l + summation
            return(H_l)

        W_0, B_0 = self.layers[0] 
        out = []
        for k in range(self.inc_nodes):
            integrand = torch.einsum('kpq,ip->ikpq', W_0[k], x[:,k,:])
            integral = torch.trapezoid(integrand, dx = self.dx['in'][k], dim = 2)
            out.append(integral)
        sum_integral = torch.sum(torch.stack(out, dim = 1), dim = 1)
        H_1 = B_0 + sum_integral  
        x = self.relu(H_1)
        for i in range(1, len(self.layers)-1):
            x = self.relu(layer_calc(x.to(device), self.layers[i], self.dx['hidden'])) 
        return layer_calc(x, self.layers[-1], self.dx['hidden'])
    


    def regularization(self):
        
        def weight_grad(W, dx_in, dx_out):
            out_s_d2 = torch.gradient(torch.gradient(W, spacing = dx_out, dim = 3)[0], spacing = dx_out, dim = 3)[0]
            out_t_d2 = torch.gradient(torch.gradient(W, spacing = dx_in, dim = 2)[0], spacing = dx_in, dim = 2)[0]
            out_d2 = (out_s_d2 + out_t_d2)**2
            out_int = torch.trapezoid(torch.trapezoid(out_d2, dx = dx_out, dim = 3), dx = dx_in, dim = 2)
            return torch.mean(out_int, dim = [0, 1])
        
        def bias_grad(B, dx_out):
            b_d2 = torch.gradient(torch.gradient(B, spacing = dx_out, dim = 1)[0], spacing = dx_out, dim = 1)[0]**2
            b_int = torch.trapezoid(b_d2, dx = dx_out, dim = 1)
            return torch.mean(b_int)

        weight = 0
        for j in range(self.inc_nodes):
            out_t_d2 = torch.gradient(torch.gradient(self.layers[0][0][j], spacing = self.dx['in'][j], dim = 1)[0], spacing = self.dx['in'][j], dim = 1)[0]
            out_s_d2 = torch.gradient(torch.gradient(self.layers[0][0][j], spacing = self.dx['hidden'], dim = 2)[0], spacing = self.dx['hidden'], dim = 2)[0]
            out_d2 = (out_s_d2 + out_t_d2)**2
            out_int = torch.trapezoid(torch.trapezoid(out_d2, dx = self.dx['hidden'], dim = 2), dx = self.dx['in'][j], dim = 1)
            weight += torch.sum(out_int)
        weight /= self.inc_nodes
        bias = bias_grad(B = self.layers[0][1], dx_out = self.dx['hidden'])

        for l in range(1, len(self.layers)-1):
            bias += bias_grad(B = self.layers[l][1], dx_out = self.dx['hidden'])
            weight += weight_grad(W = self.layers[l][0], dx_in = self.dx['hidden'], dx_out = self.dx['hidden'])
        bias += bias_grad(B = self.layers[-1][1], dx_out = self.dx['out'])
        weight += weight_grad(W = self.layers[-1][0], dx_in = self.dx['hidden'], dx_out = self.dx['out'])
        
        return self.lambda_weight/len(self.layers) * weight + self.lambda_bias/len(self.layers) * bias



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
    def __init__(self, in_dim = 24, hidden_dim = [10,10], out_dim = 24, activation = F.relu, model_version = 'advanced'):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # num_classes : number of classes, if regression - 1 (default), for classification, set to K classes
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.activation = activation
        self.model_version = model_version
        dim = [in_dim] + hidden_dim + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        if model_version == 'simple':
            pass
        elif model_version == 'advanced':
            self.ln = nn.ModuleList([LayerNorm(k) for k in hidden_dim])
        else:
            print('Please correctly specify model version: "simple" or "advanced"')

    def forward(self, t):
        if self.model_version == 'simple':     
            for i in range(len(self.layers)-1):
                t = self.layers[i](t)
                t = self.activation(t)
        elif self.model_version == 'advanced':
            for i in range(len(self.layers)-1):
                t = self.layers[i](t)
                t = t + self.ln[i](t)  # skipping connection
                t = self.activation(t)
        return self.layers[-1](t) # linear activation at the last layer



class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, model_version = "advanced", device = None):
        super().__init__() 
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.FF = FeedForward(in_dim, hidden_dim, out_dim, model_version = model_version).to(device)
        
    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        return self.FF(x.to(self.device)).reshape(-1,1,self.out_dim)
    


class ConvLayers(nn.Module):
    def __init__(self, in_dim = 100, hidden_channels = [4,4], kernel_convolution = 8, kernel_pool = 4, convolution_stride = 1, pool_stride = 2):
        """
        in_dim               : dimensions of the functional covariate
        hidden_channels      : list of convolutional layer channels
        kernel_convolution   : length of kernel for conv1d
        kernel_pool          : length of kernel for pool
        convolution_stride   : length of stride for conv1d
        pool_stride          : length of stride for pool
        """
        super().__init__()
        dim = [1] + hidden_channels
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels = dim[i-1], out_channels = dim[i], kernel_size = kernel_convolution, stride = convolution_stride) for i in range(1, len(dim))])
        self.pool = nn.MaxPool1d(kernel_size = kernel_pool, stride = pool_stride).to(device)
        
        def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
            return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        L_in = in_dim
        for _ in range(1, len(dim)):
            L_out = calculate_output_length(L_in, kernel_convolution, convolution_stride)
            L_out = calculate_output_length(L_out, kernel_pool, pool_stride)
            L_in = L_out
        self.out_d = L_in * hidden_channels[-1]
    
    def forward(self, t):
        for i in range(len(self.conv_layers)):
            t = self.pool(F.relu(self.conv_layers[i](t)))
        return t



class CNN(nn.Module):
    def __init__(self, inc_nodes = 1, in_dim = 24, out_dim = 24, conv_hidden_channels = [4,4], fc_hidden = [24, 24],
                 kernel_convolution = 8, kernel_pool = 4, convolution_stride = 1, pool_stride = 2,
                 device = None):  
        """     
        inc_nodes            : number of functional covariates
        in_dim               : dimension of input variables, scalar
        out_dim              : dimension of input variables, scalar
        conv_hidden_channels : list of channels in each conv1d layer
        fc_hidden            : list of in_d for each hidden fully connected layer
        kernel_convolution   : length of kernel for conv1d
        kernel_pool          : length of kernel for pool
        convolution_stride   : length of stride for conv1d
        pool_stride          : length of stride for pool
        """
        super().__init__()
        self.inc_nodes = inc_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.conv_layers = nn.ModuleList([ConvLayers(in_dim, conv_hidden_channels, kernel_convolution, kernel_pool, convolution_stride, pool_stride)
                                           for in_d in range(inc_nodes)]).to(device)
        self.out_d_values = [layer.out_d for layer in self.conv_layers]
        fc_in_d = sum(self.out_d_values)
        self.fc_layers = FeedForward(in_dim = fc_in_d, hidden_dim = fc_hidden, out_dim = out_dim, model_version = "advanced").to(device)
    
    def forward(self, x):
        result = []
        for i in range(self.inc_nodes):
            output = self.conv_layers[i](x[:,i,:].unsqueeze(1).to(device))
            result.append(output)
        conv_output = torch.flatten(torch.hstack(result), 1)
        return self.fc_layers(conv_output).reshape(-1,1,self.out_dim)
    


class LSTM(nn.Module):
    def __init__(self, inc_nodes = 1, lstm_hidden = [24], out_dim = 24, fc_hidden = [32, 32], num_layers = 1, bidirectional = True, device = None):
        """
        lstm_hidden          : list of hidden sizes in LSTM for each functional covariate
        fc_hidden            : list of in_d for each hidden fully connected layer
        num_layers           : number of layers in each LSTM
        bidirectional        : True/False
        dropout              : dropout probability, a float between 0.0 and 1.0
        activation           : activation function at each layer        
        """
        super().__init__()
        self.inc_nodes = inc_nodes
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.D = 2 if bidirectional == True else 1
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.ModuleList([nn.LSTM(input_size = 1, hidden_size = lstm_hidden[i], num_layers = num_layers,
                                           bidirectional = bidirectional, batch_first = True).to(device) for i in range(self.inc_nodes)])
        fc_in_d = sum(lstm_hidden) * self.D
        self.fc_layers = FeedForward(in_dim = fc_in_d, hidden_dim = fc_hidden, out_dim = out_dim, model_version = "advanced").to(device)
        self.device = device

    def forward(self, x):
        lstm_out = []
        hidden = [torch.randn(self.D * self.num_layers, len(x), dim).to(self.device) for dim in self.lstm_hidden]
        carry = [torch.randn(self.D * self.num_layers, len(x), dim).to(self.device) for dim in self.lstm_hidden]
        for i  in range(len(self.lstm)):
            lstm_out.append(self.lstm[i](x[:,i,:].unsqueeze(2).to(self.device), (hidden[i], carry[i]))[0][:,-1])
        t = torch.hstack(lstm_out)
        return self.fc_layers(t).reshape(-1,1,self.out_dim)