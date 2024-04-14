import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.linspace(-2, 4, 200)
b0, b1, b2, b3 = 0.2, 0.35, 1, 0.03
X_true = b0*np.exp(b1*np.abs(t + np.sin(2*np.pi*t))) + b2*np.sqrt(t + 3) + b3*np.abs(t - 3)**2
X = pd.read_csv('Datasets/Scalar_on_Function/Simulation/data/task 1/B1_G1/snr0.5/X/X.csv', header = None)


plt.figure(figsize=(13,7))
for index, row in X.iloc[160:180,:].iterrows():
    plt.plot(t, row, alpha = 0.65)
plt.plot(t, X_true, label = 'Expected X(t)', linewidth = 2, color = 'black')
plt.legend(fontsize="15") 
plt.savefig('Datasets/Scalar_on_Function/Simulation/data/X(t)_sample_of_20.png', dpi=600, transparent=True)


beta1 = 0.7 + 0.5*np.sin(np.pi*t) + np.cos(2*np.pi*t)
beta2 = -0.3 + np.exp(0.15*np.abs(t + np.cos(2*np.pi*t))) + 0.1*np.sin(np.pi*t)*np.sqrt(t+3)

fig, axs = plt.subplots(1, 2, figsize = (10, 6))
axs[0].plot(t, beta1, label = r'$\beta_1(t)$', color = 'xkcd:grapefruit')
axs[0].plot(t, beta1 * X_true, label = r'$\beta_1(t) \tilde{X}(t)$', color = 'xkcd:teal')
axs[0].axhline(y = 0, linestyle = '--', color = 'xkcd:steel grey', alpha = 0.5)
axs[0].legend()
axs[1].plot(t, beta2, label = r'$\beta_2(t)$', color = 'xkcd:grapefruit')
axs[1].plot(t, beta2 * X_true, label = r'$\beta_2(t) \tilde{X}(t)$', color = 'xkcd:teal')
axs[1].axhline(y = 0, linestyle = '--', color = 'xkcd:steel grey', alpha = 0.5)
axs[1].legend()
plt.savefig('Datasets/Scalar_on_Function/Simulation/data/beta_weights.png', dpi=600, transparent=True)
plt.show()