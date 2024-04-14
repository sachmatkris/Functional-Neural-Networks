import torch
from Function_on_Function.Real.Bike_Sharing.train_functions import train_lstm
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_NAME = 'LSTM'
folder_name = 'train_' + MODEL_NAME.lower() + '_bike_sharing'
hyperparameters = {'lstm_hidden1'        : tune.choice([4, 8, 12, 16, 24]),
                   'lstm_hidden2'        : tune.choice([4, 8, 12, 16, 24]),
                   'fc_hidden_layers'   : tune.choice([1, 2, 3]),
                   'fc_hidden_nodes'    : tune.choice([16, 32, 64]),
                   'num_layers'         : tune.choice([1, 2, 3]),
                   'bidirectional'      : tune.choice([True, False]),
                   'lr'                 : tune.uniform(0.001, 0.5),
                   'MODEL_NAME'         : MODEL_NAME}


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