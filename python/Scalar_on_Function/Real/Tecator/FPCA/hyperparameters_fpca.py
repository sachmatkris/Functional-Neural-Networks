import torch
from torch import nn
import numpy as np
import pandas as pd
import random

from skfda.representation.basis import  FourierBasis, BSplineBasis
from Datasets.Scalar_on_Function import Utils, Models
import json
from itertools import product
from scipy.io import arff


MODEL_NAME = 'FPCA'
folder_name = 'train_' + MODEL_NAME.lower() + '_tecator'
hyperparameters = {'n_components'            : [7, 9],
                   'data_basis_type'         : ['fourier', 'bspline'],
                   'data_basis_num'          : [9, 11, 15, 19],
                   'component_basis_type'    : ['fourier', 'bspline'],
                   'component_basis_num'     : [9, 11, 15, 19],
                   'lr'                      : [0.001, 0.1],
                   'data_directory'          : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/',
                   'MODEL_NAME'              : MODEL_NAME}

data = pd.DataFrame(arff.loadarff(hyperparameters['data_directory'] + 'tecator.arff')[0]).iloc[:215,:]
X = data.iloc[:,:100].values
Y = torch.tensor(np.array(data.loc[:,'fat']).reshape([-1,1]), dtype=torch.float)
T = np.linspace(850, 1050, 100)

cv_folds = Utils.kfold_cv(X)
structure =  {'func': [[0, 100]], 'scalar': [100, 100]}
loss = nn.MSELoss()

n_components = hyperparameters['n_components']
lr =  np.round(np.random.uniform(hyperparameters['lr'][0], hyperparameters['lr'][1], 1000), 3)
data_component_combinations = list(product(hyperparameters['data_basis_type'], hyperparameters['data_basis_num'],
                                           hyperparameters['component_basis_type'], hyperparameters['component_basis_num'],
                                           lr))
random_samples = random.sample(data_component_combinations, 100)

EPOCHS = 300
NUM_ITER = 3
results_temp = np.zeros([NUM_ITER, 5, len(n_components) + len(n_components) * len(random_samples)])
for i in range(NUM_ITER):
    for fold_idx in range(len(cv_folds)):
        model_idx = 0
        for n_component in n_components:    
            data = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
            train_dataloader, test_dataloader = Utils.raw_fpca(data, [n_component])
            model = Models.NN(in_d = n_component, sub_hidden = [32, 32], dropout = 0, device = 'cuda', model_version = 'advanced')
            results_temp[i, fold_idx, model_idx] = Utils.pytorch_trainer(model, 'NN', loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = 0.05, device = 'cuda')
            model_idx += 1
            for parameters in random_samples:
                if parameters[0] == 'bspline':
                    data_basis = BSplineBasis([850, 1050], parameters[1])
                elif parameters[0] == 'fourier':
                    data_basis = FourierBasis([850, 1050], parameters[1])
                if parameters[2] == 'bspline':
                    component_basis = BSplineBasis([850, 1050], parameters[3])
                elif parameters[2] == 'fourier':
                    component_basis = FourierBasis([850, 1050], parameters[3])
                    
                train_dataloader, test_dataloader = Utils.basis_fpca(data,  [data_basis], [component_basis], [n_component])
                model = Models.NN(in_d = n_component, sub_hidden = [32, 32], dropout = 0, device = 'cuda', model_version = 'advanced')
                results_temp[i, fold_idx, model_idx] = Utils.pytorch_trainer(model, 'NN', loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = parameters[4], device = 'cuda')
                model_idx += 1
                print(model_idx)



model_idx = 0 
results = {}
for n_component in n_components:
    results[f'raw_{n_component}_lr0.05'] = results_temp.mean(1).mean(0)[model_idx]
    model_idx += 1
    for parameters in random_samples:
        results[f'fpca_{n_component}_data_{parameters[0]}_{parameters[1]}_component_{parameters[2]}_{parameters[3]}_lr_{parameters[4]}'] = results_temp.mean(1).mean(0)[model_idx]
        model_idx += 1
        
with open(hyperparameters['data_directory'] + hyperparameters['MODEL_NAME'] + f"/results.json",'w') as f:
    json.dump(results, f, indent = 2)

print(json.dumps(results,sort_keys=True, indent = 4))