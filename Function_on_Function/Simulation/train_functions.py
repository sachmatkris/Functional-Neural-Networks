import torch
from torch import nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Function_on_Function import Models, Utils
from ray import train
from skfda.representation.basis import  FourierBasis, BSplineBasis


class QuadraticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        shape = 1/(inputs.shape[2] - 1)
        loss = torch.mean(torch.trapezoid((inputs - targets)**2, dx = shape, dim = 2)) # same as MSE
        return loss


def train_ffbnn(config):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()

    if config['in_base'] == 'fourier':
        base1 = FourierBasis(n_basis = config['in_base_n'])
    elif config['in_base'] == 'bspline':
        base1 = BSplineBasis(n_basis = config['in_base_n'])
    if config['hidden_base'] == 'fourier':
        hidden_base = FourierBasis(n_basis = config['hidden_base_n'])
    elif config['hidden_base'] == 'bspline':
        hidden_base = BSplineBasis(n_basis = config['hidden_base_n'])
    bases = {'input' : [base1], 'hidden' : hidden_base}

    structure = {'inc_nodes' : 1, 'dimensions' : 100}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5))

    for fold_idx in range(len(cv_folds)):
        train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model = Models.FFBNN(bases = bases, q = structure['dimensions'],
                        inc_nodes = structure['inc_nodes'], hidden_nodes = [config['hidden_nodes']] * config['hidden_layers'], out_nodes = 1,
                        lambda_weight = config['lambda_weight'], lambda_bias = config['lambda_bias'], device = device)    

        loss = QuadraticLoss()
        results[fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)



def train_ffdnn(config):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()

    structure = {'inc_nodes' : 1, 'dimensions' : 100}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5))

    for fold_idx in range(len(cv_folds)):
        train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model = Models.FFDNN(inc_nodes = structure['inc_nodes'], hidden_nodes = [config['hidden_nodes']] * config['hidden_layers'], out_nodes = 1,
                                q = {'in' : structure['inc_nodes'] * [structure['dimensions']], 'hidden' : config['hidden_q'], 'out' : structure['dimensions']},
                                lambda_weight = config['lambda_weight'], lambda_bias = config['lambda_bias'], device = device)    

        loss = QuadraticLoss()
        results[fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)



def train_nn(config):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()

    structure = {'inc_nodes' : 1, 'dimensions' : 100}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5))
    for fold_idx in range(len(cv_folds)):
        train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model = Models.NN(in_dim = structure['inc_nodes'] * structure['dimensions'], hidden_dim = [config['hidden_nodes']] * config['hidden_layers'],
                          out_dim = structure['dimensions'], device = device)
        loss = QuadraticLoss()
        results[fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)



def train_cnn(config):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    
    structure = {'inc_nodes' : 1, 'dimensions' : 100}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5))
    for fold_idx in range(len(cv_folds)):
        train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model = Models.CNN(inc_nodes = structure['inc_nodes'], in_dim = structure['dimensions'], out_dim = structure['dimensions'], 
                            conv_hidden_channels = [config['conv_hidden_channels']] * config['conv_hidden_layers'],
                            fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'],
                            kernel_convolution = config['kernel_convolution'], kernel_pool = config['kernel_pool'],
                            convolution_stride = config['convolution_stride'], pool_stride = config['pool_stride'],
                            device = device) 
        loss = QuadraticLoss()
        results[fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)



def train_lstm(config):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()

    structure = {'inc_nodes' : 1, 'dimensions' : 100}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5))
    for fold_idx in range(len(cv_folds)):
        train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model = Models.LSTM(inc_nodes = structure['inc_nodes'], out_dim = structure['dimensions'], 
                            lstm_hidden = [config['lstm_hidden']],
                            fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'],
                            num_layers = config['num_layers'], bidirectional = config['bidirectional'],
                            device = device) 
        loss = QuadraticLoss()
        results[fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)