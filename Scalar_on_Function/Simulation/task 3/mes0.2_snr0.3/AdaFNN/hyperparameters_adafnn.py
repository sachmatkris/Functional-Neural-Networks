import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Datasets.Scalar_on_Function import Utils
from Datasets.Scalar_on_Function.Simulation.train_functions import train_adafnn

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler


MODEL_NAME = 'AdaFNN'
MES, SNR = 0.2, 0.3 
beta, g = 1, 1       # chosen and fixed for the whole task 2
folder_name = 'train_' + MODEL_NAME.lower() + f'_regressionsimulation_mes{MES}_snr{SNR}'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'n_bases'            : tune.choice([3, 4, 5, 6, 7]),
                   'bases_hidden_nodes' : tune.choice([8, 16, 32, 64]),
                   'bases_hidden_layers': tune.choice([1, 2, 3]),
                   'sub_hidden_nodes'   : tune.choice([16, 32, 64]),
                   'sub_hidden_layers'  : tune.choice([1, 2, 3]),
                   'lambda1'            : tune.uniform(0.0, 1.0),
                   'lambda2'            : tune.uniform(0.0, 1.0),
                   'lr'                 : tune.uniform(0.001, 0.05),
                   'data_directory'     : f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/task 3/B{beta}_G{g}/mes{MES}_snr{SNR}/',
                   'MODEL_NAME'         : MODEL_NAME,
                   'X_dir'              : f'X/X_beta{beta}_g{g}_snr{SNR}.csv',
                   'T_dir'              : f'T/T_beta{beta}_g{g}_snr{SNR}.csv',
                   'Y_dir'              : f'Y/Y_beta{beta}_g{g}_snr{SNR}.csv'}

if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_adafnn, {"cpu": 12, "gpu": 1})
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

result_df = Utils.load_best(save_directory, train_adafnn)