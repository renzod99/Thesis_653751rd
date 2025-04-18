import logging
import numpy as np
from scipy.linalg import block_diag
from scipy.special import roots_genlaguerre, gamma

from gpss.kernels.KernelABC import KernelComponent
from gpss.kernels.sqe import SQEKernel

logger = logging.getLogger(__name__)


class RQKernel(KernelComponent):
    hyperparameter_keys = ["alpha_rq", "l_rq", "sigma_rq"]
    def __init__(self, l_rq: float, sigma_rq: float, alpha_rq: float):
        self.N_se = 1      
        self.n_quad = 2
        self.l = l_rq  
        self.sigma = sigma_rq            
        self.alpha = alpha_rq

    def initialize(self):
        roots, weights = roots_genlaguerre(self.n_quad, self.alpha - 1)
        H_list = []
        F_list = []
        L_list = []
        Qc_list = []
        x_list = []
        P_list = []
        gamma_alpha = gamma(self.alpha)
        for i in range(self.n_quad):
            xi = roots[i]
            wi = weights[i]

            li = self.l * np.sqrt(self.alpha / xi)
            sigma_i = self.sigma * np.sqrt(wi / gamma_alpha)
            sqe = SQEKernel(li, sigma_i)
            init_dict = sqe.initialize()
            # print(f"{i}, li: {li}, sigma_i: {sigma_i}")
            # print(f"{init_dict["F"][-1,:]}")
            H_list.append(init_dict["H"])
            F_list.append(init_dict["F"])
            L_list.append(init_dict["L"])
            Qc_list.append(init_dict["Qc"])
            x_list.append(init_dict["x"])
            P_list.append(init_dict["P"])


        F_block = block_diag(*F_list)
        L_block = block_diag(*L_list)
        Qc_block = block_diag(*Qc_list)
        P_block = block_diag(*P_list)
        x_block = np.hstack(x_list)
        H_total = np.hstack(H_list)
        
        return {
            "H": H_total, 
            "F": F_block,
            "L": L_block, 
            "Qc": Qc_block, 
            "x": x_block, 
            "P": P_block,
        }
    
    def characteristic_coefficients(self):
        pass
    
    def calculate_Qc(self):
        pass

if __name__ == "__main__":
    ratquad = RQKernel(
        N=1,
        n_quad=6,
        length_scale=15,
        sigma=1,
        alpha=1
    )
    ratquad.initialize()
    print()