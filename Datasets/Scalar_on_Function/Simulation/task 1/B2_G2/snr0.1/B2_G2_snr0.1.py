import torch
from torch import nn
import numpy as np
import pandas as pd

from skfda import FDataGrid
from skfda.representation.basis import  FourierBasis, BSplineBasis
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix, KNeighborsHatMatrix, LocalLinearRegressionHatMatrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils



beta, g, snr = 2, 2, 0.1
data_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/Regression/B{beta}_G{g}/snr{snr}/'
save_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/Regression/B{beta}_G{g}/snr{snr}/'
Y_dir = f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'

X = pd.read_csv(data_directory + 'X/X.csv', header = None).values
T = pd.read_csv(data_directory + 'T/T.csv', header = None).values
Y = torch.from_numpy(pd.read_csv(data_directory + Y_dir, header = None).values).float()
cv_folds = Utils.kfold_cv(X)
structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
loss = nn.MSELoss()

# here we store results into (iter, FOLD, MODELS) array
NUM_ITER = 10
EPOCHS = 300
results = np.zeros(shape = (NUM_ITER, 5, 8))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')

        # NN
        in_d = (structure['func'][-1][1] - structure['func'][0][0]) + (structure['scalar'][1] - structure['scalar'][0])
        train_dataloader_nn, test_dataloader_nn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'NN', batch_size = 16)
        model_NN = Models.NN(in_d = in_d, sub_hidden = [64, 64, 64], dropout = 0, device = device)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_NN, 'NN', loss, 'regression', train_dataloader_nn, test_dataloader_nn, EPOCHS, lr = 0.02, device = 'cuda:0')

        # CNN
        train_dataloader_cnn, test_dataloader_cnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'CNN', batch_size = 16)
        model_CNN = Models.CNN(structure = structure, conv_hidden_channels = [8, 8], fc_hidden = [16, 16],
                            kernel_convolution = 8, kernel_pool = 4, convolution_stride = 1, pool_stride = 2,
                            dropout = 0, device = device)
        results[i, fold_idx, 1] = Utils.pytorch_trainer(model_CNN, 'CNN', loss, 'regression', train_dataloader_cnn, test_dataloader_cnn, EPOCHS, lr = 0.02, device = 'cuda:0')

        # LSTM
        train_dataloader_lstm, test_dataloader_lstm = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'LSTM', batch_size = 16)
        model_LSTM = Models.LSTM(structure = structure, lstm_hidden = [200],
                            fc_hidden = [32, 32], num_layers = 2, bidirectional = True,
                            dropout = 0, device = device)
        results[i, fold_idx, 2] = Utils.pytorch_trainer(model_LSTM, 'LSTM', loss, 'regression', train_dataloader_lstm, test_dataloader_lstm, EPOCHS, lr = 0.001, device = 'cuda:0')

        # FNN
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN(structure = structure, functional_bases = [BSplineBasis(n_basis = 7)],
                            phi_bases = [FourierBasis(n_basis = 13)], sub_hidden = [64, 64],
                            dropout = 0, device = device)
        results[i, fold_idx, 3] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'regression', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.001, device = 'cuda:0')

        # AdaFNN
        train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'AdaFNN', batch_size = 16)
        model_AdaFNN = Models.AdaFNN(structure = structure, n_bases = [7], bases_hidden = [[64]], sub_hidden = [16, 16, 16],
                            lambda1 = 0.7, lambda2 = 0.0, dropout = 0, device = device)     
        results[i, fold_idx, 4] = Utils.pytorch_trainer(model_AdaFNN, 'AdaFNN', loss, 'regression', train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.04, device = 'cuda:0')

        # FPCA
        data_FPCA = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
        n_component = 7
        train_dataloader_fpca, test_dataloader_fpca = Utils.raw_fpca(data_FPCA, [n_component])
        #train_dataloader_fpca, test_dataloader_fpca = Utils.basis_fpca(data_FPCA,  [BSplineBasis(n_basis = 9)], [BSplineBasis(n_basis = 11)], [n_component])
        model_FPCA = Models.NN(in_d = n_component, sub_hidden = [32, 32], dropout = 0, device = 'cuda', model_version = 'advanced')
        results[i, fold_idx, 5] = Utils.pytorch_trainer(model_FPCA, 'NN', loss, 'regression', train_dataloader_fpca, test_dataloader_fpca, EPOCHS, lr = 0.08, device = 'cuda')

        # FLM
        data_FLM = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
        results[i, fold_idx, 6] = Utils.flm(data_FLM, [BSplineBasis([-2, 4], n_basis = 7)], [BSplineBasis([-2, 4], n_basis = 5)])

        # FNLM
        train_indices, test_indices = cv_folds[fold_idx]['train'], cv_folds[fold_idx]['test']
        y_train, y_test = Y[train_indices], Y[test_indices]
        X_train = FDataGrid(X[:, structure['func'][0][0] : structure['func'][0][1]][train_indices,:], T)
        X_test = FDataGrid(X[:, structure['func'][0][0] : structure['func'][0][1]][test_indices,:], T)
        #basis = BSplineBasis(n_basis = 5)
        #X_train_basis, X_test_basis = X_train.to_basis(basis), X_test.to_basis(basis)
        results[i, fold_idx, 7] = Utils.fnlm(X_train, X_test, y_train.numpy(), y_test.numpy(), NadarayaWatsonHatMatrix(bandwidth = 0.37))
        #results[i, fold_idx, 7] = Utils.fnlm(X_train_basis, X_test_basis, y_train, y_test, LocalLinearRegressionHatMatrix(bandwidth = 0.78))

results.mean(1).mean(0)
np.savetxt(save_directory + "results.csv", results.mean(1), delimiter=",")