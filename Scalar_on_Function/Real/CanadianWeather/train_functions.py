import torch
import json
from torch import nn
import numpy as np
import pandas as pd
from scipy.io import arff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import LabelEncoder
from Datasets.Scalar_on_Function import Models, Utils
from ray import train
from skfda.representation.basis import  FourierBasis, BSplineBasis





def train_nn(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    in_d = (structure['func'][-1][1] - structure['func'][0][0]) + (structure['scalar'][1] - structure['scalar'][0])
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 4)
            model = Models.NN(in_d = in_d, sub_hidden = [config['hidden_nodes']] * config['hidden_layers'], dropout = 0, num_classes = 4, device = device)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)





def train_cnn(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 4)
            model = Models.CNN(structure = structure,
                    conv_hidden_channels = [config['conv_hidden_channels']] * config['conv_hidden_layers'],
                    fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                    kernel_convolution = config['kernel_convolution'], kernel_pool = config['kernel_pool'],
                    convolution_stride = config['convolution_stride'], pool_stride = config['pool_stride'],
                    num_classes = 4, dropout = 0, device = device)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)



def train_lstm(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 4)
            model = Models.LSTM(structure = structure, lstm_hidden = [config['lstm_hidden1'], config['lstm_hidden2']],
                                fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                                num_layers = config['num_layers'], bidirectional = config['bidirectional'],
                                num_classes = 4, dropout = 0, device = device)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)



def train_fnn_o(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    if config['weight_basis1'] == 'bspline':
        phi_base1 = BSplineBasis([0, 1], config['weight_basis_num1'])
    elif config['weight_basis1'] == 'fourier':
        phi_base1 = FourierBasis([0, 1], config['weight_basis_num1'])
    if config['weight_basis2'] == 'bspline':
        phi_base2 = BSplineBasis([0, 1], config['weight_basis_num2'])
    elif config['weight_basis2'] == 'fourier':
        phi_base2 = FourierBasis([0, 1], config['weight_basis_num2'])

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 4)
            model = Models.FNN(structure = structure, phi_bases = [phi_base1, phi_base2], 
                               sub_hidden = [config['hidden_nodes']] * config['hidden_layers'],
                               num_classes = 4, dropout = 0, device = device, smoothed = False)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, 'FNN', loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)



def train_fnn_s(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    if config['weight_basis1'] == 'bspline':
        phi_base1 = BSplineBasis([0, 1], config['weight_basis_num1'])
    elif config['weight_basis1'] == 'fourier':
        phi_base1 = FourierBasis([0, 1], config['weight_basis_num1'])
    if config['weight_basis2'] == 'bspline':
        phi_base2 = BSplineBasis([0, 1], config['weight_basis_num2'])
    elif config['weight_basis2'] == 'fourier':
        phi_base2 = FourierBasis([0, 1], config['weight_basis_num2'])

    if config['data_basis1'] == 'bspline':
        functional_base1 = BSplineBasis([0, 1], config['data_basis_num1'])
    elif config['data_basis1'] == 'fourier':
        functional_base1 = FourierBasis([0, 1], config['data_basis_num1'])
    if config['data_basis2'] == 'bspline':
        functional_base2 = BSplineBasis([0, 1], config['data_basis_num2'])
    elif config['data_basis2'] == 'fourier':
        functional_base2 = FourierBasis([0, 1], config['data_basis_num2'])

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 4)
            model = Models.FNN(structure = structure, 
                            functional_bases = [functional_base1, functional_base2], phi_bases = [phi_base1, phi_base2], 
                            sub_hidden = [config['hidden_nodes']] * config['hidden_layers'],
                            num_classes = 4, dropout = 0, device = device)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)





def train_adafnn(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 4)
            model = Models.AdaFNN(structure = structure, n_bases = [config['n_bases1'], config['n_bases2']],
                                bases_hidden = [[config['bases_hidden_nodes1']] * config['bases_hidden_layers1'],
                                                [config['bases_hidden_nodes2']] * config['bases_hidden_layers2']],
                                sub_hidden = [config['sub_hidden_nodes']] * config['sub_hidden_layers'],
                                lambda1 = config['lambda1'], lambda2 = config['lambda2'], 
                                num_classes = 4, dropout = 0, device = device)     

            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)