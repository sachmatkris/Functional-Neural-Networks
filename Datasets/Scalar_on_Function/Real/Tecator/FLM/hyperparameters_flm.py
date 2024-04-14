import numpy as np
import pandas as pd
from skfda.representation.basis import  FourierBasis, BSplineBasis
import json
from Datasets.Scalar_on_Function import Utils
import torch
import json
from itertools import product
from scipy.io import arff

MODEL_NAME = 'FLM'
save_directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/' + MODEL_NAME
hyperparameters = {'data_basis_type'      : ['bspline', 'fourier'],
                   'data_basis_num'       : [5, 7, 9],
                   'coef_basis_type'      : ['bspline', 'fourier'],
                   'coef_basis_num'       : [5, 7, 9],
                   'data_directory'       : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/',
                   'MODEL_NAME'           : MODEL_NAME}

data = pd.DataFrame(arff.loadarff(hyperparameters['data_directory'] + 'tecator.arff')[0]).iloc[:215,:]
X = data.iloc[:,:100].values
Y = torch.tensor(np.array(data.loc[:,'fat']).reshape([-1,1]), dtype=torch.float)
T = np.linspace(850, 1050, 100)

cv_folds = Utils.kfold_cv(X)
structure =  {'func': [[0, 100]], 'scalar': [100, 100]}

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
            
with open(save_directory + f"/results.json",'w') as f:
    json.dump(results, f, indent = 2)

print(json.dumps(results,sort_keys=True, indent = 4))