import numpy as np
from gpss.kernels.KernelABC import KernelComponent


class ConstantKernel(KernelComponent):
    hyperparameter_keys = []
    def __init__(self):
        self.dim = 1
    
    def initialize(self):
        H = np.array([1])
        F = np.array([[0]])
        L = np.array([[1]])
        Qc = self.calculate_Qc()
        x0 = np.array([0])
        P0 = np.array([[1]])
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x0, "P": P0}
    
    def characteristic_coefficients(self):
        return np.array([0])
    
    def calculate_Qc(self):
        return np.array([0])