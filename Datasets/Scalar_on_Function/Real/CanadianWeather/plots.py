import numpy as np
import matplotlib.pyplot as plt
import json


directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/CanadianWeather/'
json_loaded = open(directory +'CanadianWeather.json')
list_loaded = json.load(json_loaded)

original_data = np.array(list_loaded['dailyAv'])
X = original_data[:,:,[0,2]].swapaxes(1,0).reshape(35,-1, order = 'F') # only temperature and log10 precipitation
Y = np.array(list_loaded['region'])
t = np.linspace(1, 365, 365)

color_dict = {'Arctic'      : 'xkcd:ocean blue',
              'Atlantic'    : 'xkcd:crimson',
              'Continental' : 'xkcd:irish green',
              'Pacific'     : 'xkcd:golden yellow'}
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12,10), height_ratios=[3, 2]) 
for i in range(35):
    axs[0].plot(t, X[i, :365], c = color_dict[Y[i]], label = Y[i], alpha = 0.85)
    axs[1].plot(t, X[i, 365:], c = color_dict[Y[i]], alpha = 0.65)
axs[0].set_xlabel('Day')
axs[1].set_xlabel('Day')
axs[0].set_ylabel('Temperature')
axs[1].set_ylabel(r'$\log_{10}$-precipitation')
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[0].legend(by_label.values(), by_label.keys())
plt.savefig('Datasets/Scalar_on_Function/Real/CanadianWeather/canadianweather_data.png', dpi=400, transparent=True)