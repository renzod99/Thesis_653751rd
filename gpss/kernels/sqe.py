import logging
import math
import sympy as sp
import autograd.numpy as np

from gpss.kernels.KernelABC import KernelComponent


logger = logging.getLogger(__name__)

class SQEKernel(KernelComponent):
    hyperparameter_keys = ["l_sqe", "sigma_sqe"]
    def __init__(self, l_sqe: float, sigma_sqe: float):
        # if method not in ('taylor', 'pade'):
        #     method = 'pade'
        self.method = 'pade'
        if self.method == 'pade':
            self.base_n = 1
            self.n_states = 4 * self.base_n
        else:
            self.n_states = 6
        self.l = l_sqe
        self.sigma = sigma_sqe

    def initialize(self):
        H = np.zeros(self.n_states)
        F = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states - 1):
            F[i, i+1] = 1
        
        if self.method == 'taylor':
            H[0] = 1
            d = self.taylor_characteristic_coefficients()
            F[-1, :] = d
        if self.method == 'pade':
            num, denom = self.pade_characteristic_coefficients()
            for i, val in enumerate(num[:-1]):
                H[i] = val
            F[-1, :] = denom


        L= np.zeros((self.n_states, 1))
        L[-1, 0] = 1

        Qc = self.calculate_Qc()

        x = np.zeros(self.n_states)
        P = np.eye(self.n_states)
        return {"H": H, "F": F, "L": L, "Qc": Qc, "x": x, "P": P}

    def characteristic_coefficients(self):
        if self.method == 'taylor':
            coeffs = self.taylor_characteristic_coefficients()
            return coeffs
        else:
            a, b = self.pade_characteristic_coefficients()
            return (a,b)

    def calculate_Qc(self):
        if self.method == 'taylor':
            kappa = (1/(2*self.l**2))
            q = self.sigma**2 * np.factorial(self.n_states) * (4 * kappa )** self.n_states * np.sqrt(np.pi / kappa)
            return q
        else:
            c = np.sqrt(2 * np.pi)
            q = self.sigma**2 * self.l * c
        return q

    def pade_characteristic_coefficients(self):
        L, M = 2*self.base_n, 4*self.base_n
        
        a = [
            (-1)**j * sp.factorial(L + M - j) * sp.factorial(M) /
            (sp.factorial(L + M) * sp.factorial(j) * sp.factorial(M - j))
            for j in range(1, M + 1)
        ]
        a = np.insert(a, 0, 1)
        
        b = [
            sp.factorial(L + M - j) * sp.factorial(L) /
            (sp.factorial(L + M) * sp.factorial(j) * sp.factorial(L - j))
            for j in range(L + 1)
        ]
    
        s, l, omega = sp.symbols('s l om')
        x = (-(l**2 * s**2) / 2) 
        a = [a[j] * x**j for j in range(len(a))]
        b = [b[j] * x**j for j in range(len(b))]
        a = [expr.subs(l, self.l) for expr in a]
        b = [expr.subs(l, self.l) for expr in b]
        
        a_poly_spec = sp.Poly(sum(a), s)
        b_poly_spec = sp.Poly(sum(b), s)
        
        a_roots = a_poly_spec.nroots(maxsteps=200)
        stable_a = [root for root in a_roots if sp.re(root) < 0]
        transfer_func_denom = sp.prod((omega - root) for root in stable_a)
        transfer_func_denom = sp.poly(sp.expand(transfer_func_denom), omega)
        # print(transfer_func_denom)
        denom_coeffs = transfer_func_denom.all_coeffs()
        tol = 1e-8
        denom_coeffs = np.array([float(c.evalf(n=50).as_real_imag()[0]) if abs(c.evalf().as_real_imag()[1]) < tol else complex(c.evalf()) for c in denom_coeffs])
        denom_coeffs = -np.flip(denom_coeffs[1:])

        b_roots = b_poly_spec.nroots(maxsteps=200)
        stable_b = [root for root in b_roots if sp.re(root) < 0]
        transfer_func_num = sp.prod((omega - root) for root in stable_b)
        transfer_func_num = sp.poly(sp.expand(transfer_func_num), omega)
        # print(transfer_func_num)
        num_coeffs = transfer_func_num.all_coeffs()
        num_coeffs = np.array([float(c.evalf(n=50).as_real_imag()[0]) if abs(c.evalf().as_real_imag()[1]) < tol else complex(c.evalf()) for c in num_coeffs])
        num_coeffs = np.flip(num_coeffs)
        return num_coeffs, denom_coeffs

    def taylor_characteristic_coefficients(self):            
        x = sp.Symbol('x')
        kappa = 1 / (2 * self.l**2)
        P = 0
        for n in range(self.n_states + 1):
            term = (math.factorial(self.n_states) / math.factorial(n)) * ((-1) ** n) * ((4 * kappa) ** (self.n_states - n)) * (x ** (2 * n))
            P += term
        polynomial = sp.poly(P, x)
        roots = polynomial.nroots(maxsteps=200)
        stable_roots = [root for root in roots if sp.re(root) < 0]
        transfer_func = sp.prod((x - root) for root in stable_roots)
        transfer_func = sp.poly(sp.expand(transfer_func), x)
        coefficients = transfer_func.all_coeffs()
        tol = 1e-8
        coeffs = np.array([float(c.evalf(n=50).as_real_imag()[0]) if abs(c.evalf().as_real_imag()[1]) < tol else complex(c.evalf()) for c in coefficients])
        coeffs = np.flip(coeffs[1:])
        return -coeffs