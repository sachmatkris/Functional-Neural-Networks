import torch
from Function_on_Function.Real.Bike_Sharing.train_functions import train_cnn
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'CNN'
folder_name = 'train_' + MODEL_NAME.lower() + '_bike_sharing'
hyperparameters = {'conv_hidden_layers'     : tune.choice([1, 2, 3]),
                   'conv_hidden_channels'   : tune.choice([4, 8, 16]),
                   'fc_hidden_layers'       : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'        : tune.choice([16, 32, 64]),
                   'kernel_convolution'     : tune.choice([2, 3, 4, 5, 6]),
                   'kernel_pool'            : tune.choice([2, 3, 4]),
                   'convolution_stride'     : tune.choice([1]),
                   'pool_stride'            : tune.choice([1]),
                   'lr'                     : tune.uniform(0.001, 0.5),
                   'MODEL_NAME'             : MODEL_NAME}


if __name__ == "__main__":
    sched = AsyncHyperBandScheduler()
    trainable_with_cpu_gpu = tune.with_resources(train_cnn, {"cpu": 12, "gpu": 1})
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