import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from sklearn.preprocessing import StandardScaler
from skfda.preprocessing.dim_reduction import FPCA
from ray import tune


class QuadraticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        shape = 1/(inputs.shape[2] - 1)
        loss = torch.mean(torch.trapezoid((inputs - targets)**2, dx = shape, dim = 2)) # same as MSE
        return loss


def kfold_cv(data, k = 5):
    random_indices = np.arange(len(data))
    np.random.seed(5)
    np.random.shuffle(random_indices)
    sets = np.array_split(random_indices, k)
    folds = []
    for i in range(k):
        train_indices = np.hstack(sets[:i] + sets[i+1:])
        test_indices = sets[i]
        folds.append({'train':train_indices, 'test':test_indices})
    return folds


def get_data_loaders(structure, data_in, data_out, cv_folds, fold_idx, batch_size = 16):
    train_indices, test_indices = cv_folds[fold_idx]['train'], cv_folds[fold_idx]['test']
    data_input_train, data_input_test = data_in[train_indices,:], data_in[test_indices,:]
    y_train, y_test = data_out[train_indices].reshape(-1, 1, structure['dimensions']), data_out[test_indices].reshape(-1, 1, structure['dimensions'])
    train, test = data_input_train.reshape(-1, structure['inc_nodes'], structure['dimensions']), data_input_test.reshape(-1, structure['inc_nodes'], structure['dimensions'])

    train_dataset = TensorDataset(torch.from_numpy(train).float(), y_train)
    test_dataset = TensorDataset(torch.from_numpy(test).float(), y_test)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False) 

    return train_dataloader, test_dataloader


def pytorch_trainer(model, model_name, loss, train_dataloader, test_dataloader, EPOCHS, early_stop_patience = 50, lr = 0.05, device = 'cuda:0'):
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience = 50, factor = 0.5)
    early_stop_patience = early_stop_patience 
    epochs_without_improvement = 0
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        for X, y in train_dataloader:
            y_pred = model(X)
            if model_name in ['NN', 'CNN', 'LSTM']:
                batch_loss = torch.sqrt(loss(y_pred, y.to(device)))
            elif model_name in ['FFDNN', 'FFBNN']:
                batch_loss = torch.sqrt(loss(y_pred, y.to(device)) + model.regularization())
            else:
                print('No such model')
                break
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
        test_loss = 0      
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_pred = model(X)
                if model_name in ['NN', 'CNN', 'LSTM']:
                    batch_loss = torch.sqrt(loss(test_pred, y.to(device)))
                elif model_name in ['FFDNN', 'FFBNN']:
                    batch_loss = torch.sqrt(loss(test_pred, y.to(device)))
                else:
                    print('No such model')
                    break
                test_loss += batch_loss
            test_loss /= len(test_dataloader)
        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check for early stopping
        if epochs_without_improvement >= early_stop_patience:
            break
    return(best_loss)

                

def pytorch_trainer_model(model, model_name, loss, task, train_dataloader, test_dataloader, EPOCHS, early_stop_patience = 50, lr = 0.05, device = 'cuda:0'):
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience = 50, factor = 0.5)
    early_stop_patience = early_stop_patience 
    epochs_without_improvement = 0
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        for X, y in train_dataloader:
            y_pred = model(X)
            if model_name in ['NN', 'CNN', 'LSTM']:
                batch_loss = torch.sqrt(loss(y_pred, y.to(device)))
            elif model_name in ['FFDNN', 'FFBNN']:
                batch_loss = torch.sqrt(loss(y_pred, y.to(device)) + model.regularization())
            else:
                print('No such model')
                break
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
        test_loss = 0      
        with torch.inference_mode():
            for X, y in test_dataloader:
                test_pred = model(X)
                batch_loss = torch.sqrt(loss(test_pred, y.to(device)))
                test_loss += batch_loss
            test_loss /= len(test_dataloader)
        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check for early stopping
        if epochs_without_improvement >= early_stop_patience:
            break
        return (model, best_loss)




def load_best(directory, model):
    restored_tuner = tune.Tuner.restore(directory, trainable = model)
    result_grid = restored_tuner.get_results()
    return result_grid.get_dataframe()