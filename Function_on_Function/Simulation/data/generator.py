import numpy as np
import pandas as pd
from pathlib import Path
import os
# Data Generator

g = 3
beta = 1
mes = 0.2
snr = 0.1   # snr = 0.2 g = 1; snr = 0.1  g = 2, 3
n = 300
folder = f'Function_on_Function/Simulation/data/B{beta}_G{g}/'

class SoFRDataGenerator:
    def __init__(self, grid, beta, g, mes, snr):
        self.shape = 100
        self.t, self.s = grid.reshape(self.shape, 1), grid.reshape(1, self.shape)
        self.grid_length = grid.shape[0]
        self.beta_name = beta
        self.g = g
        self.mes = mes
        self.snr = snr


    def generate(self, n = 1000):
        self.n = n
        b0 = np.random.normal(loc = 0.2, scale = 0.1, size = (n,self.shape,1))
        b1 = np.random.uniform(low = 0.2, high = 0.5, size = (n,self.shape,1))
        b2 = np.random.uniform(low = 0.7, high = 1.3, size = (n,self.shape,1))
        b3 = np.random.normal(loc = 0.03, scale = 0.02, size = (n,self.shape,1))
        self.X_true = 0.2*np.exp(0.35*np.abs(self.t + np.sin(2*np.pi*self.t))) + np.sqrt(self.t + 3) + 0.03*np.abs(self.t - 3)**2
        self.X_noiseless = b0*np.exp(b1*np.abs(self.t + np.sin(2*np.pi*self.t))) + b2*np.sqrt(self.t + 3) + b3*np.abs(self.t - 3)**2
        self.X = self.X_noiseless + np.random.normal(scale=self.mes, size=(n, self.grid_length,1))
        
        if self.beta_name == 1:
            self.beta = (-0.3 + np.exp(0.15*np.abs(self.t + np.cos(2*np.pi*self.t))) + 0.1*np.sin(np.pi*self.t)*np.sqrt(self.t+3)) @ np.ones(self.s.shape)
        elif self.beta_name == 2:
            self.beta = np.sin(self.s) + 0.1*self.t @ (self.s - 4) - 0.3 + np.exp(0.15*np.abs(self.t @ (self.s - 1) + np.cos(2*np.pi * self.t))) + 0.1*np.einsum('tR,ts->ts', np.sin(np.pi*self.t), np.nan_to_num(np.sqrt(self.t @ self.s + 3)))

        if self.g == 1:
            error = np.einsum('it,itR->it', np.random.normal(scale = self.snr, size = (n, self.grid_length)), self.X)
            self.surface = np.einsum('ts,isR->its', self.beta, self.X)
            self.Y_noiseless = np.trapz(self.surface, dx = 1/(self.shape - 1), axis = 2)
            self.Y = self.Y_noiseless + error
            
            self.true_surface = np.einsum('ts,sR ->ts', self.beta, self.X_true)
            self.true_Y = np.trapz(self.true_surface, dx = 1/(self.shape - 1), axis = 1)

        elif self.g == 2:
            error = np.einsum('it,tR->it', np.random.normal(scale = self.snr, size = (n, self.grid_length)), np.abs(self.t))
            self.surface = np.sqrt(1 + np.abs(np.einsum('ts,isR->its', self.beta, self.X)))
            self.Y_noiseless = np.trapz(self.surface, dx = 1/(self.shape - 1), axis = 2)
            self.Y = self.Y_noiseless + error

            self.true_surface = np.sqrt(1 + np.abs(np.einsum('ts,sR ->ts', self.beta, self.X_true)))
            self.true_Y = np.trapz(self.true_surface, dx = 1/(self.shape - 1), axis = 1)

        elif self.g == 3:
            error1 = np.random.normal(scale=self.snr, size=(n, self.grid_length))
            error2 = np.random.normal(scale=0.2*self.snr, size=(n, self.grid_length))
            self.surface1 = np.einsum('ts,isR->its', self.beta, self.X)
            self.surface2 = 3 - np.sqrt(np.abs(np.einsum('ts,isR->its', self.beta, self.X) + 2))
            self.Y_noiseless1 = np.trapz(self.surface1, dx = 1/(self.shape - 1), axis = 2)
            self.Y_noiseless2 = np.trapz(self.surface2, dx = 1/(self.shape - 1), axis = 2)
            self.Y = np.where(self.X.mean(1).reshape(-1,1) < 2.5, self.Y_noiseless1 + error1, self.Y_noiseless2 + error2)
        
            self.true_surface1= np.einsum('ts,sR ->ts', self.beta, self.X_true)
            self.true_Y1 = np.trapz(self.true_surface1, dx = 1/(self.shape - 1), axis = 1)
            self.true_surface2 = 3 - np.sqrt(np.abs(np.einsum('ts,sR->ts', self.beta, self.X_true) + 2))
            self.true_Y2 = np.trapz(self.true_surface2, dx = 1/(self.shape - 1), axis = 1)
        else:
            print('There is no such version for a non-linearity g(t)')


            
    def save(self, folder = 'Function_Function/Simulation/data/task 1/'):
        """
        folder : folder where observations are saved
        """
        if not os.path.exists(folder + 'X/'):
            os.makedirs(folder + 'X/')
        if not os.path.exists(folder + 'Y/'):
            os.makedirs(folder + 'Y/')
        if not os.path.exists(folder + 'T/'):
            os.makedirs(folder + 'T/')
        if not os.path.exists(f'Function_on_Function/Simulation/data/beta{self.beta_name}/'):
            os.makedirs(f'Function_on_Function/Simulation/data/beta{self.beta_name}/')
        
    
        Path(folder).mkdir(parents=True, exist_ok=True)
        X_df = pd.DataFrame(self.X.squeeze(2))
        Y_df = pd.DataFrame(self.Y)
        T_df = pd.DataFrame(self.t.reshape((1, -1)))
        beta = pd.DataFrame(self.beta)
        X_df.to_csv(folder + f"X/X_beta{self.beta_name}_g{self.g}_snr{self.snr}.csv", index=False, header=None)
        Y_df.to_csv(folder + f"Y/Y_beta{self.beta_name}_g{self.g}_snr{self.snr}.csv", index=False, header=None)
        T_df.to_csv(folder + f"T/T_beta{self.beta_name}_g{self.g}_snr{self.snr}.csv", index=False, header=None)
        beta.to_csv(f"Function_on_Function/Simulation/data/beta{self.beta_name}/beta{self.beta_name}.csv", index=False, header=None)
        
        if self.g != 3:
            if not os.path.exists(folder + 'true_surface/'):
                os.makedirs(folder + 'true_surface/')
            true_surface = pd.DataFrame(self.true_surface)
            true_surface.to_csv(folder + f"true_surface/true_surface.csv", index=False, header=None)
        else:
            if not os.path.exists(folder + 'true_surface1/'):
                os.makedirs(folder + 'true_surface1/')
            if not os.path.exists(folder + 'true_surface2/'):
                os.makedirs(folder + 'true_surface2/')
            true_surface1 = pd.DataFrame(self.true_surface1)
            true_surface2 = pd.DataFrame(self.true_surface2)
            true_surface1.to_csv(folder + f"true_surface1/true_surface1.csv", index=False, header=None)
            true_surface2.to_csv(folder + f"true_surface2/true_surface2.csv", index=False, header=None)
        

grid = np.linspace(-2, 4, 100)
obj = SoFRDataGenerator(grid, beta = beta, g = g, mes = mes, snr = snr)
obj.generate(n)
obj.save(folder)