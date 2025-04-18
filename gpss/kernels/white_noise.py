import numpy as np
from gpss.kernels.KernelABC import KernelComponent


class WhiteNoiseKernel(KernelComponent):
    hyperparameter_keys = ["sigma_wn"]
    def __init__(self, sigma_wn):
        self.variance = sigma_wn**2
    
    def initialize(self):
        H = np.array([1.0])
        F = np.array([[0.0]])
        L = np.array([[1.0]])
        Qc = self.variance
        x = np.array([0.0])
        P = np.array([[self.variance]])
        
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x, "P": P}
    
    def characteristic_coefficients(self):
        pass
    
    def calculate_Qc(self):
        return self.variance
