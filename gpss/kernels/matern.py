import logging 
import autograd.numpy as np
import sympy as sp
from autograd.scipy.special import gamma

from gpss.kernels.KernelABC import KernelComponent


logger = logging.getLogger(__name__)

class MaternKernel12(KernelComponent):
    hyperparameter_keys = ["l_matern", "sigma_matern"]
    def __init__(self, l_matern: float, sigma_matern: float):
        self.n_states = 1
        self.p = self.n_states - 1
        self.l = l_matern
        self.sigma = sigma_matern

    def initialize(self):
        H = np.zeros(self.n_states)
        H[0] = 1

        F = np.zeros((self.n_states, self.n_states))
        for i in range(self.p):
            F[i, i+1] = 1
        coeffs = self.characteristic_coefficients()
        F[-1, : ] = coeffs

        L = np.zeros((self.n_states, 1))
        L[-1, 0] = 1
        Qc = self.calculate_Qc()

        x = np.zeros(self.n_states)
        P = np.eye(self.n_states)  
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P, 
        }
    
    def characteristic_coefficients(self):
        s, lambda_symbolic = sp.symbols('s lambda')
        characteristic_poly = (lambda_symbolic + s)**(self.p+1)
        expanded_poly = sp.expand(characteristic_poly)
        coeffs = [expanded_poly.coeff(s, i) for i in range(self.p+1)]
        lmbda = np.sqrt(2 * (self.p + (1/2))) / self.l
        numerical = np.array([coeff.subs(lambda_symbolic, lmbda) for coeff in coeffs])
        return -numerical
      
    def calculate_Qc(self):
        nu = self.p + 1/2
        lam = np.sqrt(2 * nu / self.l)
        numerator = 2 * self.sigma**2 * np.sqrt(np.pi) * lam**(2*self.p + 1) * gamma(self.p + 1)
        denominator = gamma(self.p + 0.5)
        q = numerator / denominator
        Qc = q
        return Qc
   
class MaternKernel32(KernelComponent):
    hyperparameter_keys = ["l_matern", "sigma_matern"]
    def __init__(self, l_matern: float, sigma_matern: float):
        self.n_states = 2
        self.p = self.n_states - 1
        self.l = l_matern
        self.sigma = sigma_matern

    def initialize(self):
        H = np.zeros(self.n_states)
        H[0] = 1

        F = np.zeros((self.n_states, self.n_states))
        for i in range(self.p):
            F[i, i+1] = 1
        coeffs = self.characteristic_coefficients()
        F[-1, : ] = coeffs

        L = np.zeros((self.n_states, 1))
        L[-1, 0] = 1
        Qc = self.calculate_Qc()

        x = np.zeros(self.n_states)
        P = np.eye(self.n_states)  
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P, 
        }
    
    def characteristic_coefficients(self):
        s, lambda_symbolic = sp.symbols('s lambda')
        characteristic_poly = (lambda_symbolic + s)**(self.p+1)
        expanded_poly = sp.expand(characteristic_poly)
        coeffs = [expanded_poly.coeff(s, i) for i in range(self.p+1)]
        lmbda = np.sqrt(2 * (self.p + (1/2))) / self.l
        numerical = np.array([coeff.subs(lambda_symbolic, lmbda) for coeff in coeffs])
        return -numerical
      
    def calculate_Qc(self):
        nu = self.p + 1/2
        lam = np.sqrt(2 * nu / self.l)
        numerator = 2 * self.sigma**2 * np.sqrt(np.pi) * lam**(2*self.p + 1) * gamma(self.p + 1)
        denominator = gamma(self.p + 0.5)
        q = numerator / denominator
        return q
    
class MaternKernel52(KernelComponent):
    hyperparameter_keys = ["l_matern", "sigma_matern"]
    def __init__(self, l_matern: float, sigma_matern: float):
        self.n_states = 3
        self.p = self.n_states - 1
        self.l = l_matern
        self.sigma = sigma_matern

    def initialize(self):
        H = np.zeros(self.n_states)
        H[0] = 1

        F = np.zeros((self.n_states, self.n_states))
        for i in range(self.p):
            F[i, i+1] = 1
        coeffs = self.characteristic_coefficients()
        F[-1, : ] = coeffs

        L = np.zeros((self.n_states, 1))
        L[-1, 0] = 1
        Qc = self.calculate_Qc()

        x = np.zeros(self.n_states)
        P = np.eye(self.n_states)  
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P, 
        }
    
    def characteristic_coefficients(self):
        s, lambda_symbolic = sp.symbols('s lambda')
        characteristic_poly = (lambda_symbolic + s)**(self.p+1)
        expanded_poly = sp.expand(characteristic_poly)
        coeffs = [expanded_poly.coeff(s, i) for i in range(self.p+1)]
        lmbda = np.sqrt(2 * (self.p + (1/2))) / self.l
        numerical = np.array([coeff.subs(lambda_symbolic, lmbda) for coeff in coeffs])
        return -numerical
      
    def calculate_Qc(self):
        nu = self.p + 1/2
        lam = np.sqrt(2 * nu / self.l)
        numerator = 2 * self.sigma**2 * np.sqrt(np.pi) * lam**(2*self.p + 1) * gamma(self.p + 1)
        denominator = gamma(self.p + 0.5)
        q = numerator / denominator
        return q