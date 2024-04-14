import json
import torch
from torch import nn
import numpy as np

from skfda.representation.basis import  FourierBasis, BSplineBasis
from sklearn.preprocessing import LabelEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils


directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/'
json_loaded = open(directory + 'CanadianWeather.json')
list_loaded = json.load(json_loaded)
original_data = np.array(list_loaded['dailyAv'])
input_data = original_data.swapaxes(1,0).reshape(35,-1, order = 'F')   # shape is (35, 365*3) full
X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
regions = np.array(list_loaded['region'])
encoder = LabelEncoder()
encoder.fit(regions)
Y = torch.tensor(encoder.transform(regions)).long()
T = np.linspace(0, 365, 365)

cv_folds = Utils.kfold_cv(X)
structure = {'func' : [[0, 365], [365, 730]], 'scalar' : [730, 730]}
loss = nn.CrossEntropyLoss()

# here we store results into (iter, FOLD, MODELS) array
NUM_ITER = 10
EPOCHS = 500
results = np.zeros(shape = (NUM_ITER, 5, 6))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')

        # NN
        in_d = (structure['func'][-1][1] - structure['func'][0][0]) + (structure['scalar'][1] - structure['scalar'][0])
        train_dataloader_nn, test_dataloader_nn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'NN', batch_size = 16)
        model_NN = Models.NN(in_d = in_d, sub_hidden = [32, 32], num_classes = 4, dropout = 0, device = device)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_NN, 'NN', loss, 'classification', train_dataloader_nn, test_dataloader_nn, EPOCHS, lr = 0.02, device = 'cuda:0')

        # CNN
        train_dataloader_cnn, test_dataloader_cnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'CNN', batch_size = 16)
        model_CNN = Models.CNN(structure = structure, conv_hidden_channels = [4], fc_hidden = [16, 16],
                            kernel_convolution = 4, kernel_pool = 4, convolution_stride = 1, pool_stride = 2,
                            num_classes = 4, dropout = 0, device = device)
        results[i, fold_idx, 1] = Utils.pytorch_trainer(model_CNN, 'CNN', loss, 'classification', train_dataloader_cnn, test_dataloader_cnn, EPOCHS, lr = 0.02, device = 'cuda:0')

        # LSTM
        train_dataloader_lstm, test_dataloader_lstm = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'LSTM', batch_size = 16)
        model_LSTM = Models.LSTM(structure = structure, lstm_hidden = [10, 10],
                            fc_hidden = [64], num_layers = 1, bidirectional = True,
                            num_classes = 4, dropout = 0, device = device)
        results[i, fold_idx, 2] = Utils.pytorch_trainer(model_LSTM, 'LSTM', loss, 'classification', train_dataloader_lstm, test_dataloader_lstm, EPOCHS, lr = 0.015, device = 'cuda:0')

        # FNN
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN(structure = structure, functional_bases = [BSplineBasis(n_basis = 15), BSplineBasis(n_basis = 15)],
                            phi_bases = [FourierBasis(n_basis = 13), BSplineBasis(n_basis = 13)], sub_hidden = [32, 32, 32],
                            num_classes = 4, dropout = 0, device = device)
        results[i, fold_idx, 3] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'classification', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.025, device = 'cuda:0')

        # AdaFNN
        train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'AdaFNN', batch_size = 16)
        model_AdaFNN = Models.AdaFNN(structure = structure, n_bases = [4, 3], bases_hidden = [[16, 16], [64]], sub_hidden = [32, 32],
                            lambda1 = 0.3, lambda2 = 0.5, num_classes = 4, dropout = 0, device = device)     
        results[i, fold_idx, 4] = Utils.pytorch_trainer(model_AdaFNN, 'AdaFNN', loss, 'classification', train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.05, device = 'cuda:0')

        # FPCA
        data_FPCA = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
        n_component = [7, 7]
        #train_dataloader_fpca, test_dataloader_fpca = Utils.raw_fpca(data_FPCA, [n_component])
        train_dataloader_fpca, test_dataloader_fpca = Utils.basis_fpca(data_FPCA,  [BSplineBasis(n_basis = 19), BSplineBasis(n_basis = 9)], [BSplineBasis(n_basis = 19), FourierBasis(n_basis = 11)], n_component)
        model_FPCA = Models.NN(in_d = sum(n_component), sub_hidden = [32, 32], num_classes = 4, dropout = 0, device = 'cuda', model_version = 'advanced')
        results[i, fold_idx, 5] = Utils.pytorch_trainer(model_FPCA, 'NN', loss, 'classification', train_dataloader_fpca, test_dataloader_fpca, EPOCHS, lr = 0.03, device = 'cuda')


results.mean(1).mean(0)
np.savetxt(directory + "results.csv", results.mean(1), delimiter=",")