import torch
from Datasets.Scalar_on_Function import Utils

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch import nn
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils
from ray import train
from skfda.representation.basis import  FourierBasis, BSplineBasis


MODEL_NAME = 'FNN_new'
folder_name = 'train_' + MODEL_NAME.lower() + '_canadianweather'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'weight_basis1'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num1'   : tune.choice([3, 5, 7, 9, 13, 15]),
                   'weight_basis2'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num2'   : tune.choice([3, 5, 7, 9, 13, 15]),
                   'hidden_layers'       : tune.choice([1, 2, 3]),
                   'hidden_nodes'        : tune.choice([16, 32, 64]),
                   'lr'                  : tune.uniform(0.001, 0.03),
                   'data_directory'      : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/',
                   'MODEL_NAME'          : MODEL_NAME}



def train_fnn(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_loaded = open('C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/CanadianWeather.json')
    list_loaded = json.load(json_loaded)
    
    original_data = np.array(list_loaded['dailyAv'])
    X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
    regions = np.array(list_loaded['region'])
    encoder = LabelEncoder()
    encoder.fit(regions)
    Y = torch.tensor(encoder.transform(regions)).long()
    structure = {'func':[[0,365],[365,730]], 'scalar':[730,730]}

    if config['weight_basis1'] == 'bspline':
        phi_base1 = BSplineBasis([0, 1], config['weight_basis_num1'])
    elif config['weight_basis1'] == 'fourier':
        phi_base1 = FourierBasis([0, 1], config['weight_basis_num1'])
    if config['weight_basis2'] == 'bspline':
        phi_base2 = BSplineBasis([0, 1], config['weight_basis_num2'])
    elif config['weight_basis2'] == 'fourier':
        phi_base2 = FourierBasis([0, 1], config['weight_basis_num2'])

    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 4)
            model = Models.FNN(structure = structure, phi_bases = [phi_base1, phi_base2], 
                               sub_hidden = [config['hidden_nodes']] * config['hidden_layers'],
                               num_classes = 4, dropout = 0, device = device, smoothed = False)
            loss = nn.CrossEntropyLoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, 'FNN', loss, 'classification', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"accuracy" : results.mean().item()}
    train.report(cv_loss)




if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_fnn, {"cpu": 12, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config = tune.TuneConfig(
            metric = "accuracy",
            mode = "max",
            scheduler = sched,
            num_samples = 100
        ),
        run_config = train.RunConfig(
            name = folder_name
        ),
        param_space = hyperparameters,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)










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
results = np.zeros(shape = (NUM_ITER, 5, 1))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN(structure = structure, phi_bases = [FourierBasis(n_basis = 13), FourierBasis(n_basis = 7)],
                               sub_hidden = [64, 64, 64], num_classes = 4, dropout = 0, device = device, smoothed = False)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'classification', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.008, device = 'cuda:0')

results.mean(1).mean(0)
np.savetxt(directory + "FNN/non-smoothed/results.csv", results.mean(1), delimiter=",")