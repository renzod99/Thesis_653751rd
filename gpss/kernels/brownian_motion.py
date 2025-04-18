import numpy as np
from gpss.kernels.KernelABC import KernelComponent

 
class BrownianMotion(KernelComponent):
    hyperparameter_keys = ["sigma_brown"]
    def __init__(self, sigma_brown):
        self.sigma = sigma_brown
    
    def initialize(self):
        H = np.array([1])  
        F = np.array([[0]]) 
        L = np.array([[1]]) 
        Qc = self.calculate_Qc() 
        x0 = np.array([0])  
        P0 = np.array([[0]])
        
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x0, "P": P0}
    
    def characteristic_coefficients(self):
        return np.array([0])
    
    def calculate_Qc(self):
        return np.array([[self.sigma**2]])

class BrownianMotionIntegrated(KernelComponent):
    hyperparameter_keys = ["sigma_brownintegrated"]
    def __init__(self, sigma):
        self.sigma = sigma
    
    def initialize(self):
        H = np.array([1, 0])  
        F = np.array([[0, 1], 
                      [0, 0]])  
        L = np.array([[0], 
                      [1]])  
        Qc = self.calculate_Qc()  
        x0 = np.array([0, 0])  
        P0 = np.array([[0, 0], 
                       [0, 0]])
        
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x0, "P": P0}
    
    def characteristic_coefficients(self):
        return np.array([0, 0]) 
    
    def calculate_Qc(self):
        return np.array([[self.sigma**2]])