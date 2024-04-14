import torch
from Datasets.Scalar_on_Function import Utils
from Datasets.Scalar_on_Function.Simulation.Regression.train_functions import train_fnn

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'FNN'
beta, g, snr = 2, 2, 0.1
folder_name = 'train_' + MODEL_NAME.lower() + f'_regressionsimulation_beta{beta}_g{g}_snr{snr}'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'weight_basis'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num'   : tune.choice([3, 5, 7, 9, 13]),
                   'data_basis'         : tune.choice(['fourier', 'bspline']),
                   'data_basis_num'     : tune.choice([5, 7, 9, 11, 15, 21]),
                   'hidden_layers'      : tune.choice([1, 2, 3]),
                   'hidden_nodes'       : tune.choice([16, 32, 64]),
                   'lr'                 : tune.uniform(0.001, 0.03),
                   'data_directory'         : f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Simulation/data/Regression/B{beta}_G{g}/snr{snr}/',
                   'MODEL_NAME'             : MODEL_NAME,
                   'Y_dir'                  : f'Y/Y_beta{beta}_g{g}_snr{snr}.csv'
                   }


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_fnn, {"cpu": 12, "gpu": 1})
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

result_df = Utils.load_best(save_directory, train_fnn)