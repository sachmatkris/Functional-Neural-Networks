import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm



data = pd.read_csv('C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Function_on_Function/Real/Bike_Sharing/hour.csv', index_col = 0)
bike_data = data.loc[data['weekday'] == 6, :] #We only consider Saturday's as in the original paper
bike_df = bike_data.pivot(index=['dteday'], columns=['hr'], values=['temp', 'hum', 'casual']).reset_index()
bike_df.columns = ['_'.join(map(str, col)) if col[0] in ['temp', 'hum', 'casual'] else col[0] for col in bike_df.columns]
data_input = bike_df.ffill().drop(['dteday'], axis = 1) # missing values filled
temp, hum, cnt = data_input.iloc[:, :24].values, data_input.iloc[:,24:48].values, data_input.iloc[:,48:].values
time = np.linspace(0, 24, 24)



np.random.seed(3)
sample = np.random.choice(len(cnt), 30)


cmap = plt.get_cmap('jet')
norm = plt.Normalize(cnt.mean(1).min(), cnt.mean(1).max())    
m = cm.ScalarMappable(cmap=cmap)
m.set_array(cnt.mean(1))
fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 10))
for index in sample:
    axs[0].plot(time, cnt[index, :], color = cmap(norm(cnt.mean(1)[index].item())), alpha = 0.65)
    axs[1].plot(time, temp[index, :], color = cmap(norm(cnt.mean(1)[index].item())), alpha = 0.65)
    axs[2].plot(time, hum[index, :], color = cmap(norm(cnt.mean(1)[index].item())), alpha = 0.65)
axs[0].set_xlabel('Hour')
axs[0].set_ylabel('Count (casual users)')
axs[1].set_xlabel('Hour')
axs[1].set_ylabel('Temperature')
axs[2].set_xlabel('Hour')
axs[2].set_ylabel('Humidity')
radius_2Dline = plt.Line2D((0, 1), (0, 0), color='k', linewidth=2)
fig.colorbar(m, ax = axs[1]).set_label('Average daily casual users ', size=10) 
fig.colorbar(m, ax = axs[2]).set_label('Average daily casual users ', size=10) 
fig.savefig('Datasets/Function_on_Function/Real/Bike_Sharing/bike_sharing.png', dpi=600, transparent=True)