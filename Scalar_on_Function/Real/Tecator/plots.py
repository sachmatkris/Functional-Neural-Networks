import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import matplotlib.cm as cm


directory = f'C:/Users/Kristijonas/Desktop/ETH/Master thesis/Datasets/Scalar_on_Function/Real/Tecator/'
data = pd.DataFrame(arff.loadarff(directory + 'tecator.arff')[0]).iloc[:215,:]
X = data.iloc[:,:100]
Y = np.array(data.loc[:,'fat']).reshape([-1,1])
t = np.linspace(850, 1050, 100)

cmap = plt.get_cmap('viridis_r')
norm = plt.Normalize(Y.min(), Y.max())    
m = cm.ScalarMappable(cmap=cmap)
m.set_array(Y.squeeze())
fig, ax = plt.subplots(figsize = (13,7)) 
for index, row in X.iloc[:50,:].iterrows():
    ax.plot(t, row, color=cmap(norm(Y[index].item())), alpha = 0.65)
ax.set_xlabel('Wavelength (in nm)')
ax.set_ylabel('Absorbance')
radius_2Dline = plt.Line2D((0, 1), (0, 0), color='k', linewidth=2)
fig.colorbar(m, ax = ax).set_label('Fat', size=15) 
plt.savefig('Datasets/Scalar_on_Function/Real/Tecator/tecator_data.png', dpi=400, transparent=True)