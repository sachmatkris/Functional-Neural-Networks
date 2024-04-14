import torch
from Datasets.Scalar_on_Function import Utils

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch import nn
import numpy as np
import pandas as pd
from scipy.io import arff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils
from ray import train
from skfda.representation.basis import  FourierBasis, BSplineBasis


MODEL_NAME = 'FNN_new'
folder_name = 'train_' + MODEL_NAME.lower() + '_tecator'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'weight_basis'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num'   : tune.choice([3, 5, 7, 9, 13, 15]),
                   'hidden_layers'      : tune.choice([1, 2, 3]),
                   'hidden_nodes'       : tune.choice([16, 32, 64]),
                   'lr'                 : tune.uniform(0.001, 0.03),
                   'data_directory'     : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/'}

def train_fnn(config):
    EPOCHS = 500
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.DataFrame(arff.loadarff(config['data_directory'] + 'tecator.arff')[0]).iloc[:215,:]
    X = data.iloc[:,:100].values
    Y = torch.tensor(np.array(data.loc[:,'fat']).reshape([-1,1]), dtype=torch.float)
    structure = {'func' : [[0, 100]], 'scalar' : [100, 100]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))

    if config['weight_basis'] == 'bspline':
        phi_base = BSplineBasis([0, 1], config['weight_basis_num'])
    elif config['weight_basis'] == 'fourier':
        phi_base = FourierBasis([0, 1], config['weight_basis_num'])

    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
            model = Models.FNN_new(structure = structure, phi_bases = [phi_base], sub_hidden = [config['hidden_nodes']] * config['hidden_layers'],
                                    dropout = 0, device = device, smoothed = False)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, 'FNN', loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)




if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_fnn, {"cpu": 12, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config = tune.TuneConfig(
            metric = "mse",
            mode = "min",
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






directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/'
data = pd.DataFrame(arff.loadarff(directory + 'tecator.arff')[0]).iloc[:215,:]
X = data.iloc[:,:100].values
Y = torch.tensor(np.array(data.loc[:,'fat']).reshape([-1,1]), dtype=torch.float)
T = np.linspace(850, 1050, 100)

cv_folds = Utils.kfold_cv(X)
structure = {'func' : [[0, 100]], 'scalar' : [100, 100]}
loss = nn.MSELoss()



# here we store results into (iter, FOLD, MODELS) array
NUM_ITER = 10
EPOCHS = 500
results = np.zeros(shape = (NUM_ITER, 5, 1))
for i in range(NUM_ITER):
    print(f'Iteration no. {i}')
    for fold_idx in range(len(cv_folds)):
        print(f'Fold no. {fold_idx}')
        train_dataloader_fnn, test_dataloader_fnn = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, 'FNN', batch_size = 16)
        model_FNN = Models.FNN_new(structure = structure, phi_bases = [FourierBasis(n_basis = 15)], sub_hidden = [32],
                                   dropout = 0, device = device, smoothed = False)
        results[i, fold_idx, 0] = Utils.pytorch_trainer(model_FNN, 'FNN', loss, 'regression', train_dataloader_fnn, test_dataloader_fnn, EPOCHS, lr = 0.009, device = 'cuda:0')

results.mean(1).mean(0)
np.savetxt(directory + "FNN/non-smoothed/results.csv", results.mean(1), delimiter=",")