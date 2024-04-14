import pandas as pd
import os

common_dir = 'Datasets/Scalar_on_Function/Simulation/task 1/'

index = pd.Index(['NN', 'CNN', 'LSTM', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM', 'FNLM'])
columns = pd.MultiIndex.from_product([['β1(t)', 'β2(t)'], ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']])
final_results = pd.DataFrame(index=index, columns=columns)

for beta in range(1, 3):
    for g in range(1, 5): 
        extra_path = os.listdir(common_dir + f'B{beta}_G{g}')[0] 
        experiment_results = pd.read_csv(common_dir + f'B{beta}_G{g}/' + extra_path + f'/results.csv', header=None)
        experiment_results.columns = ['NN', 'CNN', 'LSTM', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM', 'FNLM']
        rounding = 3 if experiment_results.mean(axis=None) > 2 else 4
        mean = experiment_results.mean().round(rounding - 1).astype(str).tolist()
        std = experiment_results.std().round(rounding).astype(str).tolist()
        final_results.loc[:, (f'β{beta}(t)', f'Scenario {g}')] = [f'{mean[i] + " ± " + std[i]}' for i in range(9)]

with open(common_dir + 'task_1_results.tex', 'w', encoding='utf-8') as f:
    f.write(final_results.to_latex())
final_results