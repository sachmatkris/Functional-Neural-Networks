import torch
from torch import nn
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Models, Utils

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.train import Result


MODEL_NAME = 'LSTM'
folder_name = 'train_' + MODEL_NAME.lower() +'_regressionsimulation_beta1_g1'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'lstm_hidden'            : tune.choice([10, 25, 50, 100, 150, 200]),
                   'fc_hidden_layers'       : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'        : tune.choice([16, 32, 64]),
                   'num_layers'             : tune.choice([1, 2, 3]),
                   'bidirectional'          : tune.choice([True, False]),
                   'lr'                     : tune.uniform(0.001, 0.1),
                   'data_directory'         : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/Regression/B1_G1/',
                   'MODEL_NAME'             : MODEL_NAME
                   }


def train_model(config):
    EPOCHS = 300
    NUM_ITER = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = pd.read_csv(config['data_directory'] + 'X/X.csv', header = None).values
    #T = pd.read_csv(config['data_directory'] + 'T/T.csv', header = None).values.squeeze()
    Y = torch.from_numpy(pd.read_csv(config['data_directory'] + 'Y/Y_beta1_g1_snr0.5.csv', header = None).values).float()
    structure = {'func' : [[0, 200]], 'scalar' : [200, 200]}
    cv_folds = Utils.kfold_cv(X)
    results = np.zeros(shape = (NUM_ITER, 5))
    for i in range(NUM_ITER):
        for fold_idx in range(len(cv_folds)):
            train_dataloader, test_dataloader = Utils.get_data_loaders(structure, X, Y, cv_folds, fold_idx, config['MODEL_NAME'], batch_size = 16)
            model = Models.LSTM(structure = structure, lstm_hidden = [config['lstm_hidden']],
                                fc_hidden = [config['fc_hidden_nodes']] * config['fc_hidden_layers'], 
                                num_layers = config['num_layers'], bidirectional = config['bidirectional'],
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
    assert not results.errors

result_df = Utils.load_best(save_directory, train_model)
result_df.sort_values(['mse']).iloc[:,14:20]
best_result: Result = result_df.get_best_result()
best_result.metrics_dataframe