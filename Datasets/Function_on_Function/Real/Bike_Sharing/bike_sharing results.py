import pandas as pd

common_dir = 'Datasets/Function_on_Function/Real/Bike_Sharing/'

experiment_results = pd.read_csv(common_dir + 'results.csv', header=None).T
experiment_results.index = ['NN', 'CNN', 'LSTM', 'FFDNN', 'FFBNN']
mean_values = experiment_results.T.mean().round(2)
std_dev_values = experiment_results.T.std().round(2)
summary_df = pd.DataFrame({'Quadratic loss': mean_values, 'SD': std_dev_values})

with open(common_dir + 'bike_sharing_results.tex', 'w', encoding='utf-8') as f:
    f.write(summary_df.to_latex())