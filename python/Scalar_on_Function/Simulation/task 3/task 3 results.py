import pandas as pd

common_dir = 'Datasets/Scalar_on_Function/Simulation/task 3/'

index = pd.Index(['NN', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM'])
columns = pd.MultiIndex.from_product([['0.2', '0.8'], ['0.3', '0.5', '1']])
final_results = pd.DataFrame(index=index, columns=columns)
final_results.index.names = ['ε_mes(t)$','$ε_1$']

for mes in [0.2, 0.8]:
    for snr in [0.3, 0.5, 1]:
        experiment_results = pd.read_csv(common_dir + f'mes{mes}_snr{snr}/results.csv', header=None)
        experiment_results.columns = ['NN', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM']
        rounding = 3 if experiment_results.mean(axis=None) > 2 else 4
        mean = experiment_results.mean().round(rounding - 1).astype(str).tolist()
        std = experiment_results.std().round(rounding).astype(str).tolist()
        final_results.loc[:, (f'{mes}', f'{snr}')] = [f'{mean[i] + " ± " + std[i]}' for i in range(6)]

with open(common_dir + 'task_3_results.tex', 'w', encoding='utf-8') as f:
    f.write(final_results.to_latex())
final_results