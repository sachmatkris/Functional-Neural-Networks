import torch
from torch import nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils
from ray import train
from skfda.representation.basis import  FourierBasis, BSplineBasis

def train_nn(config):
    EPOCHS = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    in_d = (structure['func'][-1][1] - structure['func'][0][0]) + (structure['scalar'][1] - structure['scalar'][0])
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (5, 5))
    for i in range(5):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.NN(in_d = in_d, sub_hidden = [config['hidden_nodes']] * config['hidden_layers'], dropout = 0, device = device)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)



def train_cnn(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.CNN(structure = structure,
                    conv_hidden_channels = [config['conv_hidden_channels']] * config['conv_hidden_layers'],
                    fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                    kernel_convolution = config['kernel_convolution'], kernel_pool = config['kernel_pool'],
                    convolution_stride = config['convolution_stride'], pool_stride = config['pool_stride'],
                    dropout = 0, device = device)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)




def train_lstm(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.LSTM(structure = structure, lstm_hidden = [config['lstm_hidden']],
                                fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                                num_layers = config['num_layers'], bidirectional = config['bidirectional'],
                                dropout = 0, device = device)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)




def train_fnn(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))

    if config['weight_basis'] == 'bspline':
        phi_base = BSplineBasis([0, 1], config['weight_basis_num'])
    elif config['weight_basis'] == 'fourier':
        phi_base = FourierBasis([0, 1], config['weight_basis_num'])

    if config['data_basis'] == 'bspline':
        functional_base = BSplineBasis([0, 1], config['data_basis_num'])
    elif config['data_basis'] == 'fourier':
        functional_base = FourierBasis([0, 1], config['data_basis_num'])

    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.FNN(structure = structure, 
                            functional_bases = [functional_base], phi_bases = [phi_base], 
                            sub_hidden = [config['hidden_nodes']] * config['hidden_layers'], dropout = 0, device = device)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)




def train_adafnn(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + config['X_dir'], header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))

    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.AdaFNN(structure = structure, n_bases = [config['n_bases']],
                                bases_hidden = [[config['bases_hidden_nodes']] * config['bases_hidden_layers']],
                                sub_hidden = [config['sub_hidden_nodes']] * config['sub_hidden_layers'],
                                lambda1 = config['lambda1'], lambda2 = config['lambda2'], dropout = 0, device = device)     

            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)