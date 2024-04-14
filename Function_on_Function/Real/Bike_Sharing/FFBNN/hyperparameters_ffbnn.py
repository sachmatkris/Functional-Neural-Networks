import torch
from Function_on_Function.Real.Bike_Sharing.train_functions import train_ffbnn
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'FFBNN'
folder_name = 'train_' + MODEL_NAME.lower() + '_bike_sharing'
hyperparameters = {'in_base1'           : tune.choice(['fourier', 'bspline']),
                   'in_base2'           : tune.choice(['fourier', 'bspline']),
                   'hidden_base'        : tune.choice(['fourier', 'bspline']),
                   'in_base1_n'         : tune.choice([5, 9, 15, 21]),
                   'in_base2_n'         : tune.choice([5, 9, 15, 21]),
                   'hidden_base_n'      : tune.choice([5, 7, 9, 15]),
                   'hidden_layers'      : tune.choice([1, 2, 3]),
                   'hidden_nodes'       : tune.choice([8, 16, 32, 64]), 
                   'lambda_weight'      : tune.uniform(0.0, 1.0),
                   'lambda_bias'        : tune.uniform(0.0, 1.0),
                   'lr'                 : tune.uniform(0.001, 0.5),
                   'MODEL_NAME'         : MODEL_NAME}


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_ffbnn, {"cpu": 12, "gpu": 1})
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