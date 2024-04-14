import torch
from Scalar_on_Function.Simulation.train_functions import train_lstm

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'LSTM'
task = 1
beta, g, snr = 1, 4, 0.5
folder_name = 'train_' + MODEL_NAME.lower() + f'_regressionsimulation_beta{beta}_g{g}_snr{snr}_task{task}'
hyperparameters = {'lstm_hidden'            : tune.choice([10, 25, 50, 100, 150, 200]),
                   'fc_hidden_layers'       : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'        : tune.choice([16, 32, 64]),
                   'num_layers'             : tune.choice([1, 2, 3]),
                   'bidirectional'          : tune.choice([True, False]),
                   'lr'                     : tune.uniform(0.001, 0.1),
                   'data_directory'         : f'Scalar_on_Function/Simulation/data/task {task}/B{beta}_G{g}/snr{snr}/',
                   'MODEL_NAME'             : MODEL_NAME,
                   'Y_dir'                  : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'
                   }


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_lstm, {"cpu": 12, "gpu": 1})
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