import torch
from torch import nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler


MODEL_NAME = 'CNN'
beta, g, snr = 1, 4, 0.5
folder_name = 'train_' + MODEL_NAME.lower() + f'_regressionsimulation_beta{beta}_g{g}_snr{snr}'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'conv_hidden_layers'     : tune.choice([1, 2, 3]),
                   'conv_hidden_channels'   : tune.choice([4, 8, 16]),
                   'fc_hidden_layers'       : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'        : tune.choice([16, 32, 64]),
                   'kernel_convolution'     : tune.choice([4, 8]),
                   'kernel_pool'            : tune.choice([2, 4]),
                   'convolution_stride'     : tune.choice([1, 2]),
                   'pool_stride'            : tune.choice([1, 2]),
                   'lr'                     : tune.uniform(0.001, 0.1),
                   'data_directory'         : f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/Regression/B{beta}_G{g}/snr{snr}/',
                   'MODEL_NAME'             : MODEL_NAME,
                   'Y_dir'                  : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'
                   }


def train_model(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + 'X/X.csv', header = None).values
    #T = pd.read_csv(config['data_directory'] + 'T/T.csv', header = None).values.squeeze()
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.CNN(structure = structure,
                    conv_hidden_channels = [config['conv_hidden_channels']] * config['conv_hidden_layers'],
                    fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                    kernel_convolution = config['kernel_convolution'], kernel_pool = config['kernel_pool'],
                    convolution_stride = config['convolution_stride'], pool_stride = config['pool_stride'],
                    dropout = 0, device = device)
            loss = nn.MSELoss()
            results[i, fold_idx] = Utils.pytorch_trainer(model, config['MODEL_NAME'], loss, 'regression', train_dataloader, test_dataloader, EPOCHS, lr = config['lr'], device = 'cuda:0')
    cv_loss = {"mse" : results.mean().item()}
    train.report(cv_loss)


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_model, {"cpu": 12, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_cpu_gpu,
        tune_config = tune.TuneConfig(
            metric = "mse",
            mode = "min",
            scheduler = sched,
            num_samples = 50,
        ),
        run_config = train.RunConfig(
            name = folder_name
        ),
        param_space = hyperparameters,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

result_df = Utils.load_best(save_directory, train_model)
result_df.sort_values(['mse']).iloc[:,14:20]