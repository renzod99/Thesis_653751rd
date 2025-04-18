import numpy as np
from gpss.kernels.KernelABC import KernelComponent


class LinearKernel(KernelComponent):
    hyperparameter_keys = []
    def __init__(self):
        self.dim = 2

    def initialize(self):
        H = np.zeros(self.dim)
        H[0] = 1
        F = np.array([[0, 1],
                      [0, 0]])
        
        L = np.zeros((self.dim, 1))
        L[-1, 0] = 1
        Qc = self.calculate_Qc()
        
        x = np.zeros(self.dim)
        P = np.array(   
            [[1, 0],
            [0, 1]])        
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P,
        }

    def characteristic_coefficients(self):
        pass

    def calculate_Qc(self):
        return np.zeros((1, 1), dtype=float)
