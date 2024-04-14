import torch
from torch import nn
import numpy as np
import pandas as pd

from skfda.representation.basis import  FourierBasis, BSplineBasis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Scalar_on_Function import Models, Utils

MES, SNR = 0.2, 0.3 
beta, g = 1, 1       # chosen and fixed for the whole task 2
save_directory = f'Scalar_on_Function/Simulation/task 3/mes{MES}_snr{SNR}/'
data_directory = f'Scalar_on_Function/Simulation/data/task 3/B{beta}_G{g}/mes{MES}_snr{SNR}/'
X_dir = f'X/X_beta{beta}_g{g}_snr{SNR}.csv'
T_dir = f'T/T_beta{beta}_g{g}_snr{SNR}.csv'
Y_dir = f'Y/Y_beta{beta}_g{g}_snr{SNR}.csv'

X = pd.read_csv(data_directory + X_dir, header = None).values
T = pd.read_csv(data_directory + T_dir, header = None).values
Y = torch.from_numpy(pd.read_csv(data_directory + Y_dir, header = None).values).float()
cv_folds = Utils.kfold_cv(X)
structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
loss = nn.MSELoss()

# here we store results into (iter, FOLD, MODELS) array
NUM_ITER = 10
EPOCHS = 300
results = np.zeros(shape = (NUM_ITER, 5, 6))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')

        # NN
        in_d = (structure['func'][-1][1] - structure['func'][0][0]) + (structure['scalar'][1] - structure['scalar'][0])
        train_dataloader_nn, test_dataloader_nn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'NN', batch_size = 16)
        model_NN = Models.NN(in_d = in_d, sub_hidden = [64, 64, 64], dropout = 0, device = device)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_NN, 'NN', loss, 'regression', train_dataloader_nn, test_dataloader_nn, EPOCHS, lr = 0.008, device = 'cuda:0')

        # FNN_o
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN(structure = structure, phi_bases = [FourierBasis(n_basis = 15)],
                               sub_hidden = [64, 64], dropout = 0, device = device, smoothed = False)
        results[i, fold_idx, 1] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'regression', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.006, device = 'cuda:0')

        # FNN_s
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN(structure = structure, functional_bases = [BSplineBasis(n_basis = 11)],
                            phi_bases = [FourierBasis(n_basis = 3)], sub_hidden = [32, 32, 32],
                            dropout = 0, device = device)
        results[i, fold_idx, 2] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'regression', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.005, device = 'cuda:0')

        # AdaFNN
        train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'AdaFNN', batch_size = 16)
        model_AdaFNN = Models.AdaFNN(structure = structure, n_bases = [6], bases_hidden = [[8, 8, 8]], sub_hidden = [64],
                            lambda1 = 0.8, lambda2 = 0.6, dropout = 0, device = device)     
        results[i, fold_idx, 3] = Utils.pytorch_trainer(model_AdaFNN, 'AdaFNN', loss, 'regression', train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.03, device = 'cuda:0')

        # FPCA
        data_FPCA = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
        n_component = 7
        train_dataloader_fpca, test_dataloader_fpca = Utils.raw_fpca(data_FPCA, [n_component])
        #train_dataloader_fpca, test_dataloader_fpca = Utils.basis_fpca(data_FPCA,  [BSplineBasis(n_basis = 15)], [BSplineBasis(n_basis = 15)], [n_component])
        model_FPCA = Models.NN(in_d = n_component, sub_hidden = [32, 32], dropout = 0, device = 'cuda', model_version = 'advanced')
        results[i, fold_idx, 4] = Utils.pytorch_trainer(model_FPCA, 'NN', loss, 'regression', train_dataloader_fpca, test_dataloader_fpca, EPOCHS, lr = 0.05, device = 'cuda')

        # FLM
        data_FLM = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
        results[i, fold_idx, 5] = Utils.flm(data_FLM, [FourierBasis([-2, 4], n_basis = 7)], [FourierBasis([-2, 4], n_basis = 7)])

results.mean(1).mean(0)
np.savetxt(save_directory + "results.csv", results.mean(1), delimiter=",")