import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.ml.regression import LinearRegression, KernelRegression
from ray import tune


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) 
    return acc

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

# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.with_parameters.html#
def get_data_loaders(structure, data_in, data_out, cv_folds, fold_idx, model_name, batch_size = 16):
    # have to pass data_in as a numpy array already, data_out as a tensor
    functional_part = [structure['func'][0][0], structure['func'][-1][1]]
    scalar_part = structure['scalar']

    train_indices, test_indices = cv_folds[fold_idx]['train'], cv_folds[fold_idx]['test']
    data_input_train, data_input_test, y_train, y_test = data_in[train_indices,:], data_in[test_indices,:], data_out[train_indices], data_out[test_indices]
    func_train_raw, func_test_raw = data_input_train[:, functional_part[0] : functional_part[1]], data_input_test[:, functional_part[0] : functional_part[1]]
    scalar_train_raw, scalar_test_raw = data_input_train[:, scalar_part[0] : scalar_part[1]], data_input_test[:, scalar_part[0] : scalar_part[1]]
    
    if model_name in ['NN', 'CNN', 'LSTM', 'AdaFNN']:
        func_scaler = StandardScaler()
        func_scaler.fit(func_train_raw)
        func_train, func_test = func_scaler.transform(func_train_raw), func_scaler.transform(func_test_raw)
    elif model_name == 'FNN':
        func_train, func_test = func_train_raw, func_test_raw
    else: 
        print('No such model')

    if scalar_part[1] - scalar_part[0] != 0:
        scalar_scaler = StandardScaler()
        scalar_scaler.fit(scalar_train_raw)
        scalar_train, scalar_test = scalar_scaler.transform(scalar_train_raw), scalar_scaler.transform(scalar_test_raw)
        train = torch.from_numpy(np.hstack([func_train, scalar_train])).float()
        test = torch.from_numpy(np.hstack([func_test, scalar_test])).float()
    else:
        train = torch.from_numpy(func_train).float()
        test = torch.from_numpy(func_test).float()
    train_dataset = TensorDataset(train, y_train)
    test_dataset = TensorDataset(test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False) 

    return train_dataloader, test_dataloader

def pytorch_trainer(model, model_name, loss, task, train_dataloader, test_dataloader, EPOCHS, early_stop_patience = 30, lr = 0.05, device = 'cuda:0'):
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=30, factor = 0.5)
    early_stop_patience = early_stop_patience 
    epochs_without_improvement = 0
    if task == 'regression': 
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            for X, y in train_dataloader:
                y_pred = model(X)
                if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                    batch_loss = loss(y_pred, y.to(device))
                elif model_name == 'AdaFNN':
                    batch_loss = loss(y_pred, y.to(device)) + model.R1() + model.R2()
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
                    if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                        batch_loss = loss(test_pred, y.to(device))
                    elif model_name == 'AdaFNN':
                        batch_loss = loss(test_pred, y.to(device))
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

    elif task == 'classification': 
        best_loss = float('inf')
        best_acc = 0
        for epoch in range(EPOCHS):
            for X, y in train_dataloader:
                y_logits = model(X)
                if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                    batch_loss = loss(y_logits, y.to(device))
                elif model_name == 'AdaFNN':
                    batch_loss = loss(y_logits, y.to(device)) + model.R1() + model.R2()
                else:
                    print('No such model')
                    break
                optim.zero_grad()
                batch_loss.backward()
                optim.step()
            test_loss = 0     
            test_acc = 0 
            with torch.inference_mode():
                for X, y in test_dataloader:
                    test_logits = model(X)
                    if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                        batch_loss = loss(test_logits, y.to(device))
                    elif model_name == 'AdaFNN':
                        batch_loss = loss(test_logits, y.to(device)) + model.R1() + model.R2()
                    else:
                        print('No such model')
                        break                        
                    test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1).to(device)
                    test_acc += accuracy_fn(y_true = y.to(device), y_pred = test_pred)
                    test_loss += batch_loss
                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                best_acc = test_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # Check for early stopping
            if epochs_without_improvement >= early_stop_patience:
                break
        return(best_acc)
                

def pytorch_trainer_model(model, model_name, loss, task, train_dataloader, test_dataloader, EPOCHS, early_stop_patience = 30, lr = 0.05, device = 'cuda:0'):
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=30, factor = 0.5)
    early_stop_patience = early_stop_patience 
    epochs_without_improvement = 0
    if task == 'regression': 
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            for X, y in train_dataloader:
                y_pred = model(X)
                if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                    batch_loss = loss(y_pred, y.to(device))
                elif model_name == 'AdaFNN':
                    batch_loss = loss(y_pred, y.to(device)) + model.R1() + model.R2()
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
                    if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                        batch_loss = loss(test_pred, y.to(device))
                    elif model_name == 'AdaFNN':
                        batch_loss = loss(test_pred, y.to(device))
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
        return (model, best_loss)

    elif task == 'classification': 
        best_loss = float('inf')
        best_acc = 0
        for epoch in range(EPOCHS):
            for X, y in train_dataloader:
                y_logits = model(X)
                if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                    batch_loss = loss(y_logits, y.to(device))
                elif model_name == 'AdaFNN':
                    batch_loss = loss(y_logits, y.to(device)) + model.R1() + model.R2()
                else:
                    print('No such model')
                    break
                optim.zero_grad()
                batch_loss.backward()
                optim.step()
            test_loss = 0     
            test_acc = 0 
            with torch.inference_mode():
                for X, y in test_dataloader:
                    test_logits = model(X)
                    if model_name in ['NN', 'CNN', 'LSTM', 'FNN']:
                        batch_loss = loss(test_logits, y.to(device))
                    elif model_name == 'AdaFNN':
                        batch_loss = loss(test_logits, y.to(device)) + model.R1() + model.R2()
                    else:
                        print('No such model')
                        break                        
                    test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1).to(device)
                    test_acc += accuracy_fn(y_true = y.to(device), y_pred = test_pred)
                    test_loss += batch_loss
                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
            scheduler.step(test_loss)
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # Check for early stopping
            if epochs_without_improvement >= early_stop_patience:
                break
        return (model, best_acc)



def get_data_functional(structure, data_in, data_out, grid, cv_folds, fold_idx):

    # data_out has to be passed as a tensor

    functional_part = structure['func']
    scalar_part = structure['scalar']

    train_indices, test_indices = cv_folds[fold_idx]['train'], cv_folds[fold_idx]['test']
    data_input_train, data_input_test, y_train, y_test = data_in[train_indices,:], data_in[test_indices,:], data_out[train_indices], data_out[test_indices]
    func_train_raw = [data_input_train[:, dim[0] : dim[1]] for dim in functional_part]
    func_test_raw =  [data_input_test[:, dim[0] : dim[1]] for dim in functional_part]

    func_train = [FDataGrid(func, grid) for func in func_train_raw]
    func_test = [FDataGrid(func, grid) for func in func_test_raw]

    if scalar_part[1] - scalar_part[0] != 0:
        scalar_train, scalar_test = data_input_train[:, scalar_part[0] : scalar_part[1]], data_input_test[:, scalar_part[0] : scalar_part[1]]
        return {'train' : {'func' : func_train, 'scalar ': scalar_train, 'y' : y_train}, 'test': {'func' : func_test, 'scalar ': scalar_test, 'y' : y_test}}
    else:
        return {'train' : {'func' : func_train, 'y' : y_train}, 'test': {'func' : func_test, 'y' : y_test}}




def raw_fpca(data, n_components, batch_size = 16):
    
    # n_components : list containing components for each functional covariate in FDataGrid format

    func_covs = len(data['train']['func'])
    fpcas = [FPCA(n_components = n_components[idx]) for idx in range(func_covs)]
    [fpcas[idx].fit(data['train']['func'][idx]) for idx in range(func_covs)]
    FPCs_train = [fpcas[idx].transform(data['train']['func'][idx]) for idx in range(func_covs)]
    FPCs_test = [fpcas[idx].transform(data['test']['func'][idx]) for idx in range(func_covs)]

    if 'scalar' in data['train'].keys():
        train = np.hstack(FPCs_train + [data['train']['scalar']])
        test = np.hstack(FPCs_test + [data['train']['scalar']])
    else:
        train = np.hstack(FPCs_train)
        test = np.hstack(FPCs_test)
    
    nn_scaler = StandardScaler()
    nn_scaler.fit(train)
    train_scaled = torch.from_numpy(nn_scaler.transform(train)).float()
    test_scaled = torch.from_numpy(nn_scaler.transform(test)).float()

    train_dataset = TensorDataset(train_scaled, data['train']['y'])
    test_dataset = TensorDataset(test_scaled, data['test']['y'])
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader





def basis_fpca(data, data_basis, component_basis, n_components, batch_size = 16):

    # n_components      : list containing components for each functional covariate in FDataGrid format
    # data_basis        : list containing basis for each functional covariate
    # component_basis   : list containing basis for each functional covariate

    func_covs = len(data['train']['func'])
    train_basis = [data['train']['func'][i].to_basis(data_basis[i]) for i in range(func_covs)]
    test_basis = [data['test']['func'][i].to_basis(data_basis[i]) for i in range(func_covs)]
    fpcas_basis = [FPCA(n_components = n_components[idx], components_basis = component_basis[idx]) for idx in range(func_covs)]
    [fpcas_basis[idx].fit(train_basis[idx]) for idx in range(func_covs)]
    FPCs_train = [fpcas_basis[idx].transform(train_basis[idx]) for idx in range(func_covs)]
    FPCs_test = [fpcas_basis[idx].transform(test_basis[idx]) for idx in range(func_covs)]

    if 'scalar' in data['train'].keys():
        train = np.hstack(FPCs_train + [data['train']['scalar']])
        test = np.hstack(FPCs_test + [data['train']['scalar']])
    else:
        train = np.hstack(FPCs_train)
        test = np.hstack(FPCs_test)

    nn_scaler = StandardScaler()
    nn_scaler.fit(train)
    train_scaled = torch.from_numpy(nn_scaler.transform(train)).float()
    test_scaled = torch.from_numpy(nn_scaler.transform(test)).float()

    train_dataset = TensorDataset(train_scaled, data['train']['y'])
    test_dataset = TensorDataset(test_scaled, data['test']['y'])
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader


def flm(data, data_basis, coef_basis):

    # data_basis        : list containing basis for each functional covariate
    # coef_basis      : list containing basis for each functional covariate

    func_covs = len(data['train']['func'])
    scalar_covs = 0
    train_basis = [data['train']['func'][i].to_basis(data_basis[i]) for i in range(func_covs)]
    test_basis = [data['test']['func'][i].to_basis(data_basis[i]) for i in range(func_covs)]
    train_dict = {f"functional_covariate{i}" : train_basis[i] for i in range(func_covs)}
    test_dict = {f"functional_covariate{i}" : test_basis[i] for i in range(func_covs)}

    if 'scalar' in data['train'].keys():
        scalar_covs = data['train']['scalar'].shape[1]
        train_scalar = {f"scalar_covariate{i}" : data['train']['scalar'][:,i] for i in range(scalar_covs)}
        test_scalar = {f"scalar_covariate{i}" : data['test']['scalar'][:,i] for i in range(scalar_covs)}
        train_dict.update(train_scalar)
        test_dict.update(test_scalar)
    else:
        pass
    train = pd.DataFrame(train_dict)
    test = pd.DataFrame(test_dict)
    
    coeficients = coef_basis + [None] * scalar_covs
    linear_reg = LinearRegression(coef_basis = coeficients)
    _ = linear_reg.fit(train, data['train']['y'].squeeze())
    return mean_squared_error(data['test']['y'].squeeze(), linear_reg.predict(test))


def fnlm(X_train, X_test, y_train, y_test, hat_matrix):
    nonlinear_reg = KernelRegression(kernel_estimator = hat_matrix)
    _ = nonlinear_reg.fit(X_train, y_train)
    return mean_squared_error(y_test, nonlinear_reg.predict(X_test))



def load_best(directory, model):
    restored_tuner = tune.Tuner.restore(directory, trainable = model)
    result_grid = restored_tuner.get_results()
    return result_grid.get_dataframe()