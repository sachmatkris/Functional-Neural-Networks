import numpy as np
import pandas as pd
from skfda.representation.basis import  FourierBasis, BSplineBasis
import json
from Scalar_on_Function import Utils
import torch
import json
from itertools import product


MODEL_NAME = 'FLM'
N_SAMPLES = 500 
beta, g, snr = 2, 1, 0.5       # chosen and fixed for the whole task 2
save_directory = f'Scalar_on_Function/Simulation/task 2/n{N_SAMPLES}/' + MODEL_NAME
hyperparameters = {'data_basis_type'      : ['bspline', 'fourier'],
                   'data_basis_num'       : [5, 7, 9],
                   'coef_basis_type'      : ['bspline', 'fourier'],
                   'coef_basis_num'       : [5, 7, 9],
                   'data_directory'       : f'Scalar_on_Function/Simulation/data/task 2/B{beta}_G{g}/n{N_SAMPLES}/',
                   'MODEL_NAME'           : MODEL_NAME,
                   'X_dir'                : f'X/X_beta{beta}_g{g}_snr{snr}.csv',
                   'T_dir'                : f'T/T_beta{beta}_g{g}_snr{snr}.csv',
                   'Y_dir'                : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'}

X = pd.read_csv(hyperparameters['data_directory'] + hyperparameters['X_dir'], header = None).values
Y = torch.from_numpy(pd.read_csv(hyperparameters['data_directory'] + hyperparameters['Y_dir'], header = None).values).float()
T = pd.read_csv(hyperparameters['data_directory'] + hyperparameters['T_dir'], header = None).values

cv_folds = Utils.kfold_cv(X)
structure =  {'func': [[0, 200]], 'scalar': [200, 200]}

all_parameters = list(product(hyperparameters['data_basis_type'], hyperparameters['data_basis_num'],
                              hyperparameters['coef_basis_type'], hyperparameters['coef_basis_num']))
#random_samples = random.sample(all_parameters, 100)


results_temp = np.zeros([5, len(all_parameters)])
for fold_idx in range(len(cv_folds)):
    model_idx = 0
    data = Utils.get_data_functional(structure, X, Y, T, cv_folds, fold_idx)
    for parameters in all_parameters:
        if parameters[0] == 'bspline':
            data_basis = BSplineBasis([-2, 4], parameters[1])
        elif parameters[0] == 'fourier':
            data_basis = FourierBasis([-2, 4], parameters[1])
        if parameters[2] == 'bspline':
            coef_basis = BSplineBasis([-2, 4], parameters[3])
        elif parameters[2] == 'fourier':
            coef_basis = FourierBasis([-2, 4], parameters[3])
        results_temp[fold_idx, model_idx] = Utils.flm(data, [data_basis], [coef_basis])
        model_idx += 1
        print(model_idx)

model_idx = 0 
results = {}
for fold_idx in range(len(cv_folds)):
    model_idx = 0
    for parameters in all_parameters:
        results[f'data_{parameters[0]}_{parameters[1]}_coef_{parameters[2]}_{parameters[3]}'] = results_temp.mean(0)[model_idx]
        model_idx += 1
            
with open(save_directory + f"/results_{N_SAMPLES}.json",'w') as f:
    json.dump(results, f, indent = 2)

print(json.dumps(results,sort_keys=True, indent = 4))