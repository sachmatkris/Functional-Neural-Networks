import torch
from torch import nn
import numpy as np
import pandas as pd

from skfda.representation.basis import  FourierBasis, BSplineBasis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Function_on_Function import Models, Utils


class QuadraticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        shape = 1/(inputs.shape[2] - 1)
        loss = torch.mean(torch.trapezoid((inputs - targets)**2, dx = shape, dim = 2)) # same as MSE
        return loss


directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Real/Bike_Sharing/'
data = pd.read_csv('C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Real/Bike_Sharing/hour.csv', index_col = 0)
bike_data = data.loc[data['weekday'] == 6, :] #We only consider Saturday's as in the original paper
bike_df = bike_data.pivot(index=['dteday'], columns=['hr'], values=['temp', 'hum', 'casual']).reset_index()
bike_df.columns = ['_'.join(map(str, col)) if col[0] in ['temp', 'hum', 'casual'] else col[0] for col in bike_df.columns]
data_input = bike_df.ffill().drop(['dteday'], axis = 1) # missing values filled
X = data_input.iloc[:,:48].values
Y = torch.from_numpy(data_input.iloc[:,48:].values).float()
structure = {'inc_nodes' : 2, 'dimensions' : 24}
cv_folds = Utils.kfold_cv(X)
results = np.zeros(shape = (5))


loss = QuadraticLoss()
NUM_ITER = 10
EPOCHS = 1000
results = np.zeros(shape = (NUM_ITER, 5, 5))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')

        # NN
        train_dataloader_nn, test_dataloader_nn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_NN = Models.NN(in_dim = structure['inc_nodes'] * structure['dimensions'], hidden_dim = [16, 16], out_dim = structure['dimensions'], device = device)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_NN, 'NN', loss, train_dataloader_nn, test_dataloader_nn, EPOCHS, lr = 0.045, device = 'cuda:0')
        
        # CNN
        train_dataloader_cnn, test_dataloader_cnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_CNN = Models.CNN(inc_nodes = structure['inc_nodes'], in_dim = structure['dimensions'], out_dim = structure['dimensions'], 
                            conv_hidden_channels = [8], fc_hidden = [64, 64],
                            kernel_convolution = 6, kernel_pool = 2, convolution_stride = 1, pool_stride = 1,
                            device = device) 
        results[i, fold_idx, 1] = Utils.pytorch_trainer(model_CNN, 'CNN', loss, train_dataloader_cnn, test_dataloader_cnn, EPOCHS, lr = 0.027, device = 'cuda:0')

        # LSTM
        train_dataloader_lstm, test_dataloader_lstm = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_LSTM = Models.LSTM(inc_nodes = structure['inc_nodes'], out_dim = structure['dimensions'], 
                                lstm_hidden = [8, 16], fc_hidden = [32], num_layers = 1, bidirectional = False,
                                device = device) 
        results[i, fold_idx, 2] = Utils.pytorch_trainer(model_LSTM, 'LSTM', loss, train_dataloader_lstm, test_dataloader_lstm, EPOCHS, lr = 0.0015, device = 'cuda:0')

        # FFDNN
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_FFDNN = Models.FFDNN(inc_nodes = structure['inc_nodes'], hidden_nodes = [64, 64, 64], out_nodes = 1,
                                q = {'in' : structure['inc_nodes'] * [structure['dimensions']], 'hidden' : 24, 'out' : 24},
                                lambda_weight = 0.6, lambda_bias = 0.6, device = device)
        results[i, fold_idx, 3] = Utils.pytorch_trainer(model_FFDNN, 'FFDNN', loss, train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.25, device = 'cuda:0')

        # FFBNN
        bases = {'input' : [FourierBasis(n_basis = 15), BSplineBasis(n_basis = 21)], 'hidden' : BSplineBasis(n_basis = 15)}
        train_dataloader_adafnn, test_dataloader_adafnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, batch_size = 16)
        model_FFBNN = model = Models.FFBNN(bases = bases, q = structure['dimensions'], inc_nodes = structure['inc_nodes'], hidden_nodes = [64, 64, 64], out_nodes = 1,
                                            lambda_weight = 0.04, lambda_bias = 0.97, device = device)     
        results[i, fold_idx, 4] = Utils.pytorch_trainer(model_FFBNN, 'FFBNN', loss, train_dataloader_adafnn, test_dataloader_adafnn, EPOCHS, lr = 0.2, device = 'cuda:0')

results.mean(1).mean(0)
np.savetxt(directory + "results.csv", results.mean(1), delimiter=",")