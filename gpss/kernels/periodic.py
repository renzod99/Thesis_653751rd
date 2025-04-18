import logging
import autograd.numpy as np
from autograd.scipy.special import gamma

from scipy.special import iv

from gpss.kernels.KernelABC import KernelComponent


logger = logging.getLogger(__name__)

class PeriodicKernel(KernelComponent):
    hyperparameter_keys = ["omega"]
    def __init__(self, omega: float):
        self.J = 6
        self.dim = self.J * 2
        self.omega = omega
    
    def initialize(self):
        H = np.zeros(self.dim)
        F = self.characteristic_coefficients()
        L = np.eye(self.dim)
        Qc = self.calculate_Qc()
        x = np.zeros(self.dim)
        P = 1000 * np.eye(self.dim , self.dim)
        
        for j in range(1, self.J + 1):
            idx = 2 * (j - 1)
            H[idx] = 1
        
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P, 
        }

    def characteristic_coefficients(self):
        F = np.zeros((self.dim, self.dim))
        for j in range(1, self.J + 1):
            idx = 2 * (j - 1)
            F[idx:idx + 2, idx:idx + 2] =   [[0, -j * self.omega],
                                            [j * self.omega,  0]]
        return F

    def calculate_Qc(self):
        Qc = np.zeros((self.dim, self.dim))
        return Qc

class PeriodicKernelLists(KernelComponent):
    hyperparameter_keys = ["omega"]
    def __init__(self, omega):
        self.J = 6
        self.omega = omega
    
    def initialize(self):
        H_list = []
        F_list = self.characteristic_coefficients()
        L_list = []
        Qc_list = self.calculate_Qc()
        x_list = []
        P_list = []
        
        for j in range(1, self.J + 1):
            H_j = np.array([1, 0])
            H_list.append(H_j)
            
            L_j = np.eye(2)
            L_list.append(L_j)
            
            x_j = np.zeros(2)
            x_list.append(x_j)
            
            P_j = 1000 * np.eye(2)
            P_list.append(P_j)

        return {
            "H": H_list, 
            "F": F_list,
            "L": L_list, 
            "Qc": Qc_list, 
            "x": x_list, 
            "P": P_list, 
        }

    def characteristic_coefficients(self):
        F_list = []
        for j in range(1, self.J + 1):
            F_j = np.array([[0.0,         -j * self.omega],
                            [ j * self.omega,  0.0      ]])
            F_list.append(F_j)
        return F_list
    
    def calculate_Qc(self):
        Qc_list = []
        for j in range(1, self.J + 1):
            Qc_j = np.zeros((2, 2))
            Qc_list.append(Qc_j)
        return Qc_list

class PeriodicKernelStochastic(KernelComponent):
    hyperparameter_keys = ["omega", "l_periodic", "sigma_periodic"]
    def __init__(self, omega: float, l_periodic: float, sigma_periodic: float):
        self.J = 6
        self.dim = self.J * 2
        self.omega = omega
        self.l = l_periodic
        self.sigma = sigma_periodic

    def initialize(self):
        H = np.zeros(self.dim)
        F = self.characteristic_coefficients()             
        L = np.eye(self.dim)
        Qc = self.calculate_Qc()      
        x = np.zeros(self.dim)
        P =  np.eye(self.dim)       

        for j in range(1, self.J + 1):
            idx = 2 * (j - 1)
            H[idx] = 1        
        return {
            "H": H, 
            "F": F,
            "L": L, 
            "Qc": Qc, 
            "x": x, 
            "P": P, 
        }
    
    def characteristic_coefficients(self):
        F = np.zeros((self.dim, self.dim))       
        for j in range(1, self.J + 1):
            idx = 2 * (j - 1)
            F[idx:idx + 2, idx:idx + 2] =   [[0,           -j * self.omega],
                                            [j * self.omega,  0          ]]
        return F
    
    def calculate_Qc(self):
        Qc = np.zeros((self.dim, self.dim))     
        q_j_squared = truncated_q_j(self.J, self.l)
        qj_squared_scaled = [self.sigma**2 * qj for qj in q_j_squared]
        for j in range(1, self.J + 1):
            idx = 2 * (j - 1)
            Qc[idx:idx + 2, idx:idx + 2] = qj_squared_scaled[j - 1] * np.eye(2)
        return Qc

class PeriodicKernelStochasticLists(KernelComponent):
    hyperparameter_keys = ["omega", "l_periodic", "sigma_periodic"]
    def __init__(self, omega: float, l_periodic: float, sigma_periodic: float):
        self.J = 6
        self.omega = omega
        self.l = l_periodic
        self.sigma = sigma_periodic

    def initialize(self):
        H_list = []
        F_list = self.characteristic_coefficients()
        L_list = []
        Qc_list = self.calculate_Qc()
        x_list = []
        P_list = []

        for j in range(1, self.J + 1):
            H_j = np.array([1, 0])
            H_list.append(H_j)
            
            L_j = np.eye(2)
            L_list.append(L_j)

            x = np.zeros(2)
            x_list.append(x)

            P_j = np.eye(2)
            P_list.append(P_j)

        return {
            "H": H_list, 
            "F": F_list,
            "L": L_list, 
            "Qc": Qc_list, 
            "x": x_list, 
            "P": P_list, 
        }

    def characteristic_coefficients(self):
        F_list = []
        for j in range(1, self.J + 1):
            F_j = np.array([[0, -j * self.omega],
                            [j * self.omega,  0]])
            F_list.append(F_j)
        return F_list

    def calculate_Qc(self):
        Qc_list = []
        q_j_squared = truncated_q_j(self.J, self.l)
        qj_squared_scaled = [self.sigma * qj for qj in q_j_squared]
        for j in range(1, self.J + 1):
            Qc = qj_squared_scaled[j - 1] * np.eye(2)
            Qc_list.append(Qc)
        return Qc_list

def truncated_q_j(J, ell):
    q_tilde = []
    for j in range(J + 1):
        sum_term = 0
        for i in range((J - j) // 2 + 1):
            term = (2 * (ell ** -2)) ** (j + 2 * i) / (gamma(i + 1) * gamma(j + i + 1))
            sum_term += term
        if j == 0:
            q_tilde.append(sum_term / 2)
        else:
            q_tilde.append(sum_term)
    return q_tilde

def optimal_q_j(J, ell):
    exp_term = np.exp(-ell ** -2)
    q_optimal = [(2 * iv(j, ell ** -2)) / exp_term for j in range(J + 1)]
    return q_optimal
