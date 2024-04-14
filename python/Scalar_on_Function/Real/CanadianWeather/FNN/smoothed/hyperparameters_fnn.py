import torch
from Datasets.Scalar_on_Function import Utils
from Datasets.Scalar_on_Function.Real.CanadianWeather.train_functions import train_fnn

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'FNN'
folder_name = 'train_' + MODEL_NAME.lower() + '_canadianweather'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'weight_basis1'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num1'   : tune.choice([3, 5, 7, 9, 13]),
                   'data_basis1'         : tune.choice(['fourier', 'bspline']),
                   'data_basis_num1'     : tune.choice([5, 7, 9, 11, 15, 21]),
                   'weight_basis2'       : tune.choice(['fourier', 'bspline']),
                   'weight_basis_num2'   : tune.choice([3, 5, 7, 9, 13]),
                   'data_basis2'         : tune.choice(['fourier', 'bspline']),
                   'data_basis_num2'     : tune.choice([5, 7, 9, 11, 15, 21]),
                   'hidden_layers'       : tune.choice([1, 2, 3]),
                   'hidden_nodes'        : tune.choice([16, 32, 64]),
                   'lr'                  : tune.uniform(0.001, 0.03),
                   'data_directory'      : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/',
                   'MODEL_NAME'          : MODEL_NAME}

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

result_df = Utils.load_best(save_directory, train_fnn)