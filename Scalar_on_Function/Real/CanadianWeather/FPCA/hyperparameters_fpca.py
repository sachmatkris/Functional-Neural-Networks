import torch
from torch import nn
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from skfda.representation.basis import  FourierBasis, BSplineBasis
from Datasets.Scalar_on_Function import Utils, Models
import json
from itertools import product


MODEL_NAME = 'FPCA'
folder_name = 'train_' + MODEL_NAME.lower() + '_tecator'
hyperparameters = {'n_components1'            : [7, 9],
                   'n_components2'            : [7, 9],
                   'data_basis_type1'         : ['fourier', 'bspline'],
                   'data_basis_type2'         : ['fourier', 'bspline'],
                   'data_basis_num1'          : [9, 11, 15, 19],
                   'data_basis_num2'          : [9, 11, 15, 19],
                   'component_basis_type1'    : ['fourier', 'bspline'],
                   'component_basis_type2'    : ['fourier', 'bspline'],
                   'component_basis_num1'     : [9, 11, 15, 19],
                   'component_basis_num2'     : [9, 11, 15, 19],
                   'lr'                       : [0.001, 0.1],
                   'data_directory'           : 'Scalar_on_Function/Real/CanadianWeather/',
                   'MODEL_NAME'               : MODEL_NAME}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
json_loaded = open(hyperparameters['data_directory'] +'CanadianWeather.json')
list_loaded = json.load(json_loaded)

original_data = np.array(list_loaded['dailyAv'])
X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
regions = np.array(list_loaded['region'])
encoder = LabelEncoder()
encoder.fit(regions)
Y = torch.tensor(encoder.transform(regions)).long()
T = np.linspace(0, 365, 365)

cv_folds = Utils.kfold_cv(X)
structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}
loss = nn.CrossEntropyLoss()




lr =  np.round(np.random.uniform(hyperparameters['lr'][0], hyperparameters['lr'][1], 1000), 3)
n_component_combinations = list(product(hyperparameters['n_components1'], hyperparameters['n_components2']))
data_component_combinations = list(product(hyperparameters['data_basis_type1'], hyperparameters['data_basis_num1'],
                                           hyperparameters['component_basis_type1'], hyperparameters['component_basis_num1'],
                                           hyperparameters['data_basis_type2'], hyperparameters['data_basis_num2'],
                                           hyperparameters['component_basis_type2'], hyperparameters['component_basis_num2'],
                                           lr))
random_samples = random.sample(data_component_combinations, 50)



EPOCHS = 300
NUM_ITER = 3
results_temp = np.zeros([NUM_ITER, 5, len(n_component_combinations) + len(n_component_combinations) * len(random_samples)])
for i in range(NUM_ITER):
    for fold_idx in range(len(cv_folds)):
        model_idx = 0
        for n_components in n_component_combinations:
            data = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
            train_dataloader, test_dataloader = Utils.raw_fpca(data, n_components)

            model = Models.NN(in_d = n_components[0] + n_components[1], sub_hidden = [32, 32], dropout = 0, num_classes = 4, device = 'cuda', model_version = 'advanced')
            results_temp[i, fold_idx, model_idx] = Utils.pytorch_trainer(model, 'NN', loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = 0.05, device = 'cuda')
            model_idx += 1
            for parameters in random_samples:
                if parameters[0] == 'bspline':
                    data_basis1 = BSplineBasis([0,365], parameters[1])
                elif parameters[0] == 'fourier':
                    data_basis1 = FourierBasis([0,365], parameters[1])
                if parameters[2] == 'bspline':
                    component_basis1 = BSplineBasis([0,365], parameters[3])
                elif parameters[2] == 'fourier':
                    component_basis1 = FourierBasis([0,365], parameters[3])
                if parameters[4] == 'bspline':
                    data_basis2 = BSplineBasis([0,365], parameters[5])
                elif parameters[4] == 'fourier':
                    data_basis2 = FourierBasis([0,365], parameters[5])
                if parameters[6] == 'bspline':
                    component_basis2 = BSplineBasis([0,365], parameters[7])
                elif parameters[6] == 'fourier':
                    component_basis2 = FourierBasis([0,365], parameters[7])

                    
                train_dataloader, test_dataloader = Utils.basis_fpca(data,  [data_basis1, data_basis2], [component_basis1, component_basis2], n_components)
                model = Models.NN(in_d = n_components[0] + n_components[1], sub_hidden = [32, 32], dropout = 0, num_classes = 4, device = 'cuda', model_version = 'advanced')
                results_temp[i, fold_idx, model_idx] = Utils.pytorch_trainer(model, 'NN', loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = parameters[8], device = 'cuda')
                model_idx += 1
                print(model_idx)



model_idx = 0 
results = {}
for n_components in n_component_combinations:
    results[f'raw_{n_components[0]}_{n_components[1]}_lr0.05'] = results_temp.mean(1).mean(0)[model_idx]
    model_idx += 1
    for parameters in random_samples:
        results[f'fpca_{n_components[0]}_{n_components[1]}_data1_{parameters[0]}_{parameters[1]}_component1_{parameters[2]}_{parameters[3]}_data2_{parameters[4]}_{parameters[5]}_component2_{parameters[6]}_{parameters[7]}_lr_{parameters[8]}'] = results_temp.mean(1).mean(0)[model_idx]
        model_idx += 1
        
with open(hyperparameters['data_directory'] + hyperparameters['MODEL_NAME'] + f"/results.json",'w') as f:
    json.dump(results, f, indent = 2)

print(json.dumps(results,sort_keys=True, indent = 4))