import numpy as np
import pandas as pd
from skfda.representation.basis import  FourierBasis, BSplineBasis
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix, KNeighborsHatMatrix, LocalLinearRegressionHatMatrix
from skfda import FDataGrid
import random
import json
from Datasets.Scalar_on_Function import Utils
from itertools import product
from scipy.io import arff

MODEL_NAME = 'FNLM'
save_directory = f'Scalar_on_Function/Real/Tecator/' + MODEL_NAME
hyperparameters = {'hat_matrix'       : ['nadarayawatson', 'locallinear', 'kneighbors'],
                   'bandwidth'        : [2, 20],
                   'k_neighbors'      : [2, 50],
                   'llr basis'        : ['fourier', 'bspline'],
                   'llr basis num'    : [5, 7, 9, 11, 15],
                   'data_directory'   : 'Scalar_on_Function/Real/Tecator/',
                   'MODEL_NAME'       : MODEL_NAME}

data = pd.DataFrame(arff.loadarff(hyperparameters['data_directory'] + 'tecator.arff')[0]).iloc[:215,:]
X = data.iloc[:,:100].values
Y = np.array(data.loc[:,'fat']).reshape([-1,1])
T = np.linspace(850, 1050, 100)

cv_folds = Utils.kfold_cv(X)
structure =  {'func': [[0, 100]], 'scalar': [100, 100]}

bw = np.round(np.random.uniform(hyperparameters['bandwidth'][0], hyperparameters['bandwidth'][1], 50), 2)
k = np.random.randint(hyperparameters['k_neighbors'][0], hyperparameters['k_neighbors'][1], 50)
all_parameters = list(product(hyperparameters['hat_matrix'], bw, k, hyperparameters['llr basis'], hyperparameters['llr basis num']))
random_samples = random.sample(all_parameters, 50)

results_temp = np.zeros([5, len(random_samples)])
cv_folds = Utils.kfold_cv(X)
for fold_idx in range(len(cv_folds)):
    model_idx = 0
    train_indices, test_indices = cv_folds[fold_idx]['train'], cv_folds[fold_idx]['test']
    y_train, y_test = Y[train_indices], Y[test_indices]
    X_train = FDataGrid(X[:, structure['func'][0][0] : structure['func'][0][1]][train_indices,:], T)
    X_test = FDataGrid(X[:, structure['func'][0][0] : structure['func'][0][1]][test_indices,:], T)

    for parameters in random_samples:
        if parameters[0] == 'nadarayawatson':
            hat_matrix = NadarayaWatsonHatMatrix(bandwidth = parameters[1])
            results_temp[fold_idx, model_idx] = Utils.fnlm(X_train, X_test, y_train, y_test, hat_matrix)

        elif parameters[0] == 'locallinear':
            if parameters[3] == 'bspline':
                basis = BSplineBasis(n_basis = parameters[4])
            elif parameters[3] == 'fourier':
                basis = FourierBasis(n_basis = parameters[4])
            X_train_basis, X_test_basis = X_train.to_basis(basis), X_test.to_basis(basis)
            hat_matrix = LocalLinearRegressionHatMatrix(bandwidth = parameters[1])
            results_temp[fold_idx, model_idx] = Utils.fnlm(X_train_basis, X_test_basis, y_train, y_test, hat_matrix)

        elif parameters[0] == 'kneighbors':
            hat_matrix = KNeighborsHatMatrix(n_neighbors = parameters[2])
            results_temp[fold_idx, model_idx] = Utils.fnlm(X_train, X_test, y_train, y_test, hat_matrix)
        model_idx += 1

results = {}
for fold_idx in range(len(cv_folds)):
    model_idx = 0
    for parameters in random_samples:
        results[f'hat_{parameters[0]}_bw_{parameters[1]}_k_{parameters[2]}_basis_{parameters[3]}_n_{parameters[4]}'] = results_temp.mean(0)[model_idx]
        model_idx += 1
            
with open(save_directory + f"/results.json",'w') as f:
    json.dump(results, f, indent = 2)

print(json.dumps(results,sort_keys=True, indent = 4))