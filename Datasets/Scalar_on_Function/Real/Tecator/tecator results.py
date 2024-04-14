import pandas as pd

common_dir = 'Datasets/Scalar_on_Function/Real/Tecator/'

experiment_results = pd.read_csv(common_dir + 'results.csv', header=None).T
experiment_results.index = ['NN', 'CNN', 'LSTM', 'FNN_o', 'FNN_s', 'AdaFNN', 'FPCA', 'FLM', 'FNLM']
mean_values = experiment_results.T.mean().round(2)
std_dev_values = experiment_results.T.std().round(2)
summary_df = pd.DataFrame({'MSPE': mean_values, 'SD': std_dev_values})

with open(common_dir + 'tecator_results.tex', 'w', encoding='utf-8') as f:
    f.write(summary_df.to_latex())