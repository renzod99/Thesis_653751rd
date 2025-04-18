import autograd.numpy as np
from gpss.kernels.KernelABC import KernelComponent


class ExponentialKernel(KernelComponent):
    hyperparameter_keys = ["l_exp", "sigma_exp"]
    def __init__(self, l_exp, sigma_exp):
        self.lengthscale = l_exp
        self.sigma = sigma_exp
    
    def initialize(self):
        H = np.ones(1)
        F = np.array([[-1 / self.lengthscale]])
        L = np.array([[1]])
        Qc = self.calculate_Qc()
        x0 = np.array([0])
        P0 = np.array([[self.sigma**2]])
        
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x0, "P": P0}
    
    def characteristic_coefficients(self):
        return np.array([-1 / self.lengthscale])
    
    def calculate_Qc(self):
        return np.array([[2 * (self.sigma)**2 / self.lengthscale]])