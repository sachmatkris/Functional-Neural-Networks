import torch
import numpy as np
import pandas as pd

from skfda.representation.basis import  FourierBasis, BSplineBasis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Function_on_Function import Models, Utils


beta, g, snr = 1, 3, 0.1

data_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Simulation/data/B{beta}_G{g}/'
save_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Simulation/task/B{beta}_G{g}/'
Y_dir = f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'

X = pd.read_csv(data_directory + f'X/X_beta{beta}_g{g}_snr{snr}.csv', header = None).values
T = pd.read_csv(data_directory + f'T/T_beta{beta}_g{g}_snr{snr}.csv', header = None).values
Y = torch.from_numpy(pd.read_csv(data_directory + f'Y/Y_beta{beta}_g{g}_snr{snr}.csv', header = None).values).float()
cv_folds = Utils.kfold_cv(X)
structure = {'inc_nodes' : 1, 'dimensions' : 100}
loss = Utils.QuadraticLoss()

# here we store results into (iter, FOLD, MODELS) array
NUM_ITER = 10
EPOCHS = 1000
results = np.zeros(shape = (NUM_ITER, 5, 5))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')

        # NN
        train_dataloader_nn, test_dataloader_nn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_NN = Models.NN(in_dim = structure['inc_nodes'] * structure['dimensions'], hidden_dim = [128, 128, 128],
                             out_dim = structure['dimensions'], device = device)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_NN, 'NN', loss, train_dataloader_nn, test_dataloader_nn, EPOCHS, lr = 0.001, device = 'cuda:0')

        # CNN
        train_dataloader_cnn, test_dataloader_cnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_CNN = Models.CNN(inc_nodes = structure['inc_nodes'], in_dim = structure['dimensions'], out_dim = structure['dimensions'], 
                               conv_hidden_channels = [16, 16], fc_hidden = [64, 64], kernel_convolution = 4, kernel_pool = 2, convolution_stride = 1, pool_stride = 1,
                               device = device)
        results[i, fold_idx, 1] = Utils.pytorch_trainer(model_CNN, 'CNN', loss, train_dataloader_cnn, test_dataloader_cnn, EPOCHS, lr = 0.1, device = 'cuda:0')

        # LSTM
        train_dataloader_lstm, test_dataloader_lstm = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_LSTM = Models.LSTM(inc_nodes = structure['inc_nodes'], out_dim = structure['dimensions'],
                                 lstm_hidden = [100], fc_hidden = [32], num_layers = 1, bidirectional = False, device = device)
        results[i, fold_idx, 2] = Utils.pytorch_trainer(model_LSTM, 'LSTM', loss, train_dataloader_lstm, test_dataloader_lstm, EPOCHS, lr = 0.03, device = 'cuda:0')

        # FFDNN
        train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_FFDNN = Models.FFDNN(inc_nodes = structure['inc_nodes'], hidden_nodes = [32, 32, 32], out_nodes = 1,
                                q = {'in' : structure['inc_nodes'] * [structure['dimensions']], 'hidden' : 10, 'out' : structure['dimensions']},
                                lambda_weight = 0.5, lambda_bias = 0.3, device = device)   
        results[i, fold_idx, 4] = Utils.pytorch_trainer(model_FFDNN, 'FFDNN', loss, train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.007, device = 'cuda:0')
        
        # FFBNN
        train_dataloader_ffbnn, test_dataloader_ffbnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        bases = {'input' : [BSplineBasis(n_basis = 9)], 'hidden' : BSplineBasis(n_basis = 15)}
        model_FFBNN = Models.FFBNN(bases = bases, q = structure['dimensions'], inc_nodes = structure['inc_nodes'], hidden_nodes = [64, 64], out_nodes = 1,
                                    lambda_weight = 0.5, lambda_bias = 0.2, device = device) 
        results[i, fold_idx, 3] = Utils.pytorch_trainer(model_FFBNN, 'FFBNN', loss, train_dataloader_ffbnn, test_dataloader_ffbnn, EPOCHS, lr = 0.02, device = 'cuda:0')



results.mean(1).mean(0)
np.savetxt(save_directory + "results.csv", results.mean(1), delimiter=",")