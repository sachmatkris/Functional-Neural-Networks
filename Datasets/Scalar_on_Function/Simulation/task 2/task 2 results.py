import pandas as pd

common_dir = 'Datasets/Scalar_on_Function/Simulation/task 2/'

index = pd.Index(['NN', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM'])
columns = pd.Index(['50', '250', '500', '1000'])
final_results = pd.DataFrame(index=index, columns=columns)

for n in [50, 250, 500, 1000]:
    experiment_results = pd.read_csv(common_dir + f'n{n}/results.csv', header=None)
    experiment_results.columns = ['NN', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM']
    rounding = 3 if experiment_results.mean(axis=None) > 2 else 4
    mean = experiment_results.mean().round(rounding - 1).astype(str).tolist()
    std = experiment_results.std().round(rounding).astype(str).tolist()
    final_results.loc[:, str(n)] = [f'{mean[i] + " ± " + std[i]}' for i in range(6)]

with open(common_dir + 'task_2_results.tex', 'w', encoding='utf-8') as f:
    f.write(final_results.to_latex())
final_results