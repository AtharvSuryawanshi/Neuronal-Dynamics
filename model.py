import numpy as np
from scipy.integrate import odeint      

class HH_model:
    def __init__(self, bifurcation_type = 'saddle_node'):
        self.bifurcation_type = bifurcation_type
        assert self.bifurcation_type in ['saddle_node', 'SNIC', 'subcritical_Hopf', 'supercritical_Hopf']

        if bifurcation_type in ['saddle-node', 'SNIC']:
            self.potassium_threshold = 'high'
        else:
            self.potassium_threshold = 'low'

        # all constant params of neurons
        # sample values only
        self.C_m  = 1.0  # capacitance
        # Maximum conductances (mS/cm²)
        if self.bifurcation_type == 'subcritical_Hopf':
            self.g_Na = 4    
            self.g_K = 4     
            self.g_L = 1
        else:
            self.g_Na = 20    
            self.g_K = 10     
            self.g_L = 8   

        # reversal potential
        self.E_Na = 60
        self.E_K  = -90 
        
        if self.potassium_threshold == 'high':  
            self.E_L = -80
            self.V_mid_n = -25
        else:
            self.E_L = -78
            self.V_mid_n = -45
        
        # Kinetic parameters for gating variables
        if self.bifurcation_type == 'subcritical_Hopf':
            self.V_mid_m = -30
            self.k_m = 7
        else:
            self.V_mid_m = -20
            self.k_m = 15
            
        self.k_n = 5


    def m_inf(self, V):
        """Steady-state value of m (sodium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_m) / self.k_m))
    
    def n_inf(self, V):
        """Steady-state value of n (potassium activation)"""
        return 1 / (1 + np.exp(-(V - self.V_mid_n) / self.k_n))
    
    def tau_n(self, V):
        """Time constant of n (potassium activation)"""
        return 1 # non saddle node 
        # return 1 / (0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))) 
    
    # def tau_m(self, V):
    #     return 0.1 + 0.4 / (1 + np.exp(-(V + 40) / 10)) 
    
    # def tau_h(self, V):
    #     return 0.07 + 0.8 / (1 + np.exp(-(V + 65) / 10))    
    
    # def h_inf(self, V):
    #     return 1 / (1 + np.exp(-(V + 40) / 10))
    
    def I_Na(self, V):
        """Sodium current"""
        return self.g_Na * self.m_inf(V) * (V - self.E_Na)
    
    def I_K(self, V, n):
        """Potassium current"""
        return self.g_K * n * (V - self.E_K)
    
    def I_L(self, V):
        """Leak current"""
        return self.g_L * (V - self.E_L)
    
    @staticmethod
    def create_step_current(t, step_time, step_duration, baseline, amplitude):
        """
        Create a step current waveform
        
        Parameters:
        -----------
        t : array-like
            Time points
        step_time : float
            Time at which step begins
        step_duration : float
            Duration of the step
        baseline : float
            Baseline current value
        amplitude : float
            Step amplitude (added to baseline)
        
        Returns:
        --------
        array-like
            Current values at each time point
        """
        I = np.ones_like(t) * baseline
        step_mask = (t >= step_time) & (t < step_time + step_duration)
        I[step_mask] = baseline + amplitude
        return I
    
    def dALLdt(self, X, t, I_ext_t):
        self.dt = 0.025
        """
        Calculate derivatives for the two state variables
        
        Parameters:
        -----------
        X : list or array
            State variables [V, n]
        t : float
            Current time
        I_ext_t : callable or array-like
            External current function or array of current values
        """
       # print('X:', X)
        V, n = X
       # print(V)
        
        # Get current value at time t
        
        if callable(I_ext_t):
            I = I_ext_t(t)
        else:
            # If I_ext_t is an array, interpolate to get current value
            idx = int(t / self.dt)  # self.dt needs to be set in simulate
            I = I_ext_t[min(idx, len(I_ext_t)-1)]
        
        # Calculate membrane potential derivative
        dVdt = (I - self.I_Na(V) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        
        # Calculate potassium gating variable derivative

        dndt = (self.n_inf(V) - n) / self.tau_n(V)
        
        return [dVdt, dndt]
    
    def simulate(self, T, dt, X0, I_ext):
        """
        Basic simulation without perturbations or special pulse handling
        
        Parameters:
        -----------
        T : float
            Total simulation time
        dt : float
            Time step
        X0 : list or array
            Initial conditions [V0, n0]
        I_ext : callable or array-like
            Either a function I(t) or array of current values
        
        Returns:
        --------
        tuple
            (time points, solution array)
        """
        self.dt = dt  # Store dt for use in dALLdt
        t = np.arange(0, T, dt)
        
        # If I_ext is already an array of correct length, use it directly
        if isinstance(I_ext, (np.ndarray, list)) and len(I_ext) != len(t):
            raise ValueError("If I_ext is an array, it must have the same length as time points")
        
        # Solve ODE system
        solution = odeint(self.dALLdt, X0, t, args=(I_ext,))
        
        return t, solution