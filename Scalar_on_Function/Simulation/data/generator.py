import numpy as np
import pandas as pd
from pathlib import Path
import os
# Data Generator

task = 3
beta, g, snr = 1, 1, 1
mes = 0.8
n = 250
folder = f'Scalar_on_Function/Simulation/data/task {task}/B{beta}_G{g}/'

class SoFRDataGenerator:
    def __init__(self, grid, x = 1, beta = 1, g = 1, mes = 1, err = 1, a_mu = 0, a_sd = 1):
        self.t = grid
        self.grid_length = grid.shape[0]
        self.h = np.array(self.t[1:] - self.t[:-1]).T
        self.x = x
        self.beta = beta
        self.g = g
        self.mes = mes
        self.err = err
        self.intercept = np.random.normal(a_mu, a_sd)

    def _inner_product(self, f1, f2, h):
        prod = f1 * f2
        if len(prod.shape) < 2:
            prod = prod.reshape((1, -1))
        res = np.matmul(prod[:, :-1] + prod[:, 1:], h) / 2
        return res
    
    def generate(self, n = 1000):
        self.n = n
        if self.x == 1:
            b0 = np.random.uniform(low = 1.5, high = 3, size = (n,1))
            b1 = np.random.uniform(low = 0.05, high = 0.4, size = (n,1))
            b2 = np.random.uniform(low = 0.03, high = 0.07, size = (n,1))
            b3 = np.random.uniform(low = -1, high = 3, size = (n,1)) 
            b4 = np.random.uniform(low = 0.6, high = 1, size = (n,1))
            self.X = b0 + b1*self.t + b2*self.t**3 + b3*np.sin(np.pi*self.t) - b4*np.cos(2*np.pi*self.t) + np.random.normal(scale=self.mes, size=(n, self.grid_length))
        elif self.x == 2:
            b0 = np.random.uniform(low = 0.3, high = 1, size = (n,1))
            b1 = np.random.normal(loc = 0.8, scale = 0.3, size = (n,1))
            b2 = np.random.uniform(low = 0.004, high = 0.02, size = (n,1))
            self.X = b0 + b1*np.sin(2*np.pi*self.t) + b2*self.t**3*np.cos(3*np.pi*self.t) + np.random.normal(scale=self.mes, size=(n, self.grid_length))
        elif self.x == 3:
            b0 = np.random.normal(loc = 0.2, scale = 0.1, size = (n,1))
            b1 = np.random.uniform(low = 0.2, high = 0.5, size = (n,1))
            b2 = np.random.uniform(low = 0.7, high = 1.3, size = (n,1))
            b3 = np.random.normal(loc = 0.03, scale = 0.02, size = (n,1))
            self.X_noiseless = b0*np.exp(b1*np.abs(self.t + np.sin(2*np.pi*self.t))) + b2*np.sqrt(self.t + 3) + b3*np.abs(self.t - 3)**2
            self.X = self.X_noiseless + np.random.normal(scale=self.mes, size=(n, self.grid_length))
        else:
            print('There is no such version for X(t)')

        if self.beta == 1:
            self.Beta = 0.7 + 0.5*np.sin(np.pi*self.t) + np.cos(2*np.pi*self.t)
        elif self.beta == 2: 
            self.Beta = -0.3 + np.exp(0.15*np.abs(self.t + np.cos(2*np.pi*self.t))) + 0.1*np.sin(np.pi*self.t)*np.sqrt(self.t+3)
        else:
            print('There is no such version for beta(t)')

        if self.g == 1:
            self.Y_noiseless = self.intercept + self._inner_product(self.X, self.Beta, self.h)
            self.Y = self.Y_noiseless + np.random.normal(scale = self.err, size = n)
        elif self.g == 2:
            self.Y_noiseless = np.log(np.abs(self.intercept + self._inner_product(self.X, self.Beta, self.h)))
            self.Y = self.Y_noiseless + np.random.normal(scale = self.err, size = n)
        elif self.g == 3:
            self.Y = 1 / (1 + np.exp(-0.1*(self._inner_product(self.X, self.Beta, self.h) + self.intercept) + np.random.normal(scale = self.err, size = n)))
        elif self.g == 4:
            alpha = 0.2
            self.Y = np.where(self.X.mean(1) < 2.5, self._inner_product(self.X, self.Beta, self.h) + self.intercept + np.random.normal(scale = self.err, size = n), alpha * (np.exp(self._inner_product(self.X, self.Beta, self.h) + self.intercept - 17) - 1) + np.random.normal(scale = 0.1*self.err, size = n))    
        else:
            print('There is no such version for a non-linearity g(t)')
            
    def save(self, task, folder = 'Scalar_on_Function/Simulation/data/Regression/'):
        """
        folder : folder where observations are saved
        """

        if task == 1:
            folder_dir = folder + f'snr{self.err}/'
            if not os.path.exists(folder_dir + 'X/'):
                os.makedirs(folder_dir + 'X/')
            if not os.path.exists(folder_dir + 'Y/'):
                os.makedirs(folder_dir + 'Y/')
            if not os.path.exists(folder_dir + 'T/'):
                os.makedirs(folder_dir + 'T/')
        
        elif task == 2:
            folder_dir = folder + f'n{self.n}/'
            if not os.path.exists(folder_dir + 'X/'):
                os.makedirs(folder_dir + 'X/')
            if not os.path.exists(folder_dir + 'Y/'):
                os.makedirs(folder_dir + 'Y/')
            if not os.path.exists(folder_dir + 'T/'):
                os.makedirs(folder_dir + 'T/')

        elif task == 3:
            folder_dir = folder + f'mes{self.mes}_snr{self.err}/'
            if not os.path.exists(folder_dir + 'X/'):
                os.makedirs(folder_dir + 'X/')
            if not os.path.exists(folder_dir + 'Y/'):
                os.makedirs(folder_dir + 'Y/')
            if not os.path.exists(folder_dir + 'T/'):
                os.makedirs(folder_dir + 'T/')
                       
        Path(folder_dir).mkdir(parents=True, exist_ok=True)
        X_df = pd.DataFrame(self.X)
        Y_df = pd.DataFrame(self.Y)
        T_df = pd.DataFrame(self.t.reshape((1, -1)))
        X_df.to_csv(folder_dir + f"X/X_beta{self.beta}_g{self.g}_snr{self.err}.csv", index=False, header=None)
        Y_df.to_csv(folder_dir + f"Y/Y_beta{self.beta}_g{self.g}_snr{self.err}.csv", index=False, header=None)
        T_df.to_csv(folder_dir + f"T/T_beta{self.beta}_g{self.g}_snr{self.err}.csv", index=False, header=None)

grid = np.linspace(-2, 4, 200)
obj = SoFRDataGenerator(grid, x = 3, beta = beta, g = g, mes = mes, err = snr)
obj.generate(n)
#np.std(obj.Y - obj.Y_noiseless)
obj.save(task, folder)