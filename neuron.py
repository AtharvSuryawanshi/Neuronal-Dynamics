import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HH_model:
    def __init__(self):
        # all constant params of neurons
        self.C_m  = 1.0 
        self.g_Na = 120.0
        self.g_K  = 36.0    
        self.g_L  = 0.3 
        self.E_Na = 50.0    
        self.E_K  = -77.0   
        self.E_L  = -54.387
        self.I = 0.0
        self.V = -65.0  

        # all constants of gating variables 
        self.V_mid_m = -40.0
        self.k_m = 15.0 
        self.V_mid_n = -55.0
        self.k_n = 30.0

    def m_inf(self, V):
        return 1 / (1 + np.exp(-(V - self.V_mid_m) / self.k_m))
    
    def n_inf(self, V):
        """Steady-state value of n (potassium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_n) / self.k_n))
    
    def tau_n(self, V):
        """Time constant of n (potassium activation)"""
        return 1 / (0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))) 
    
    def tau_m(self, V):
        return 0.1 + 0.4 / (1 + np.exp(-(V + 40) / 10)) 
    
    def tau_h(self, V):
        return 0.07 + 0.8 / (1 + np.exp(-(V + 65) / 10))    
    
    def h_inf(self, V):
        return 1 / (1 + np.exp(-(V + 40) / 10))
    
    def I_Na(self, V, m, h):    
        return self.g_Na * m**3 * h * (V - self.E_Na)
    
    def I_K(self, V, n):    
        return self.g_K  * n**4 * (V - self.E_K)
    
    def I_L(self, V):
        return self.g_L * (V - self.E_L)
    
    def I_inj(self, t):
        """Return the input current at time t."""
        return 1 if 10 <= t <= 40 else 0
    
    def dXdt(self, X, t):   
        V, m, h, n = X
        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = (self.m_inf(V) - m) / self.tau_m(V)
        dhdt = (self.h_inf(V) - h) / self.tau_h(V)
        dndt = (self.n_inf(V) - n) / self.tau_n(V)
        return dVdt, dmdt, dhdt, dndt
    
    def simulate(self, t):   
        X = odeint(self.dXdt, [-65, 0, 0, 0], t)
        return X
    
    def plot(self, t, X):
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax[0].plot(t, V, 'k')
        ax[0].set_ylabel('V (mV)')
        ax[0].set_title('Hodgkin-Huxley Neuron')
        ax[1].plot(t, ina, 'c', label='$I_{Na}$')
        ax[1].plot(t, ik, 'y', label='$I_{K}$')
        ax[1].plot(t, il, 'm', label='$I_{L}$')
        ax[1].set_ylabel('Current')
        ax[1].legend()
        ax[2].plot(t, m, 'r', label='m')
        ax[2].plot(t, h, 'g', label='h')
        ax[2].plot(t, n, 'b', label='n')
        ax[2].set_ylabel('Gating Value')
        ax[2].set_xlabel('Time (ms)')
        ax[2].legend()
        plt.show()

model = HH_model()
t = np.arange(0, 50, 0.01)  
X = model.simulate(t)
model.plot(t, X)
