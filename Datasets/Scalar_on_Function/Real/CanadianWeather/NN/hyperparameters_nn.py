import torch
from Datasets.Scalar_on_Function import Utils
from Datasets.Scalar_on_Function.Real.CanadianWeather.train_functions import train_nn

from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'NN'
folder_name = 'train_' + MODEL_NAME.lower() + '_canadianweather'
save_directory = 'C:/Users/Kristijonas/ray_results/' + folder_name
hyperparameters = {'hidden_layers'      : tune.choice([1, 2, 3]),
                   'hidden_nodes'       : tune.choice([16, 32, 64, 128]),
                   'lr'                 : tune.uniform(0.001, 0.1),
                   'data_directory'     : 'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/',
                   'MODEL_NAME'         : MODEL_NAME}

if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_nn, {"cpu": 12, "gpu": 1})
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

result_df = Utils.load_best(save_directory, train_nn)