from torch import nn
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler


MODEL_NAME = 'AdaFNN'
beta, g, snr = 1, 3, 0.5



folder_name = 'train_' + MODEL_NAME.lower() + f'_regressionsimulation_beta{beta}_g{g}_snr{snr}'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'n_bases'            : tune.choice([3, 4, 5, 6, 7]),
                   'bases_hidden_nodes' : tune.choice([8, 16, 32, 64]),
                   'bases_hidden_layers': tune.choice([1, 2, 3]),
                   'sub_hidden_nodes'   : tune.choice([16, 32, 64]),
                   'sub_hidden_layers'  : tune.choice([1, 2, 3]),
                   'lambda1'            : tune.uniform(0.0, 1.0),
                   'lambda2'            : tune.uniform(0.0, 1.0),
                   'lr'                 : tune.uniform(0.001, 0.05),
                   'data_directory'     : f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/Regression/B{beta}_G{g}/snr{snr}/',
                   'MODEL_NAME'         : MODEL_NAME,
                   'Y_dir'              : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'
                   }


def train_model(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + 'X/X.csv', header = None).values
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + config['Y_dir'], header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))

    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.AdaFNN(structure = structure, n_bases = [config['n_bases']],
                                bases_hidden = [[config['bases_hidden_nodes']] * config['bases_hidden_layers']],
                                sub_hidden = [config['sub_hidden_nodes']] * config['sub_hidden_layers'],
                                lambda1 = config['lambda1'], lambda2 = config['lambda2'], dropout = 0, device = device)     

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
            num_samples = 50
        ),
        run_config = train.RunConfig(
            name = folder_name
        ),
        param_space = hyperparameters,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

result_df = Utils.load_best(save_directory, train_model)