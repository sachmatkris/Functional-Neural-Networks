import torch
from Datasets.Function_on_Function import Utils
from Datasets.Function_on_Function.Simulation.train_functions import train_lstm
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'LSTM'
beta, g, snr = 1, 2, 0.1
folder_name = 'train_' + MODEL_NAME.lower() + f'_fof_regressionsimulation_beta{beta}_g{g}'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'lstm_hidden'       : tune.choice([10, 25, 50, 100]),
                   'fc_hidden_layers'   : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'    : tune.choice([16, 32, 64]),
                   'num_layers'         : tune.choice([1, 2, 3]),
                   'bidirectional'      : tune.choice([True, False]),
                   'lr'                 : tune.uniform(0.001, 0.5),
                   'data_directory'     : f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Simulation/data/B{beta}_G{g}/',
                   'MODEL_NAME'         : MODEL_NAME,
                   'X_dir'              : f'X/X_beta{beta}_g{g}_snr{snr}.csv',
                   'T_dir'              : f'T/T_beta{beta}_g{g}_snr{snr}.csv',
                   'Y_dir'              : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'}


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_lstm, {"cpu": 12, "gpu": 1})
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

result_df = Utils.load_best(save_directory, train_lstm)