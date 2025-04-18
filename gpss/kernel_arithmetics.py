import numpy as np
from scipy.linalg import block_diag, eigvals

from filter.kf_utility import is_stable_continuous
from gpss.kernels.KernelABC import KernelComponent

class AdditiveKernel():
    def __init__(self, components):
        self.list_of_names = [component.__class__.__name__ for component in components]
        for i, comp in enumerate(components):
            if isinstance(comp, type):  # If comp is a class instead of an instance
                raise TypeError(f"Expected an instance of KernelComponent, but got {comp} (a class)!")
        if len(self.list_of_names) <= 1:
            raise ValueError(f"unsufficient kernels to add together: #kernels={len(self.list_of_names)}")
        self.components = [component.initialize() for component in components]
    
    def initialize(self):
        H_list = [comp['H'] for comp in self.components]
        F_list = [comp['F'] for comp in self.components]
        L_list = [comp['L'] for comp in self.components]
        Qc_list = [comp['Qc'] for comp in self.components]
        x_list = [comp['x'] for comp in self.components]
        P_list = [comp['P'] for comp in self.components]

        combined_F = np.array([], dtype=np.float64).reshape(0, 0)        
        for i, F in enumerate(F_list):
            try:
                is_stable_continuous(F)
            except ValueError as e:
                raise ValueError(f"Component {i, self.list_of_names[i]} has an unstable F matrix: \n{F} \n{e}")
            combined_F = F if combined_F.size == 0 else block_diag(combined_F, F)

        else:
            combined = {
                'H': np.hstack(H_list),
                'F': combined_F,
                'L': block_diag(*L_list),
                'Qc': block_diag(*Qc_list),
                'x': np.hstack(x_list),
                'P': block_diag(*P_list),
            }
        return combined

class MultiplicativeKernel():
    def __init__(self, base, multiplier):
        self.base_component = base
        self.multiplier = multiplier

    def initialize(self):
        base = self.base_component.initialize()
        multiplier = self.multiplier.initialize()
        
        H = np.kron(base['H'], multiplier['H'])
        F = np.kron(base['F'], np.eye(multiplier['F'].shape[0])) + \
            np.kron(np.eye(base['F'].shape[0]), multiplier['F'])
        L = np.kron(base['L'], multiplier['L'])
        Qc = np.kron(base['Qc'], multiplier['Qc'])
        x = np.kron(base['x'], multiplier['x'])
        P = 1000 * np.eye(F.shape[0])
        return {
            'H': H,
            'F': F,
            'L': L,
            'Qc': Qc,
            'x': x,
            'P': P,
        }
    
class MultiplicativeKernelPeriodic():
    def __init__(self, base, periodic):
        self.base_component = base
        self.periodic = periodic

    def initialize(self):
        base = self.base_component.initialize()
        periodic = self.periodic.initialize()
        J = self.periodic.J
        
        H_list, F_list, L_list, Qc_list, x_list = [], [], [], [], []
        for j in range(J):
            H_j = np.kron(base['H'], periodic['H'][j])
            F_j = np.kron(base['F'], np.eye(periodic['F'][j].shape[0])) + \
                  np.kron(np.eye(base['F'].shape[0]), periodic['F'][j])
            L_j = np.kron(base['L'], periodic['L'][j])
            Qc_j = np.kron(base['Qc'], periodic['Qc'][j])
            x_j = np.kron(base['x'], periodic['x'][j])
        
            H_list.append(H_j)        
            F_list.append(F_j)
            L_list.append(L_j)
            Qc_list.append(Qc_j)
            x_list.append(x_j)

        H = np.hstack(H_list)
        F = block_diag(*F_list)
        L = block_diag(*L_list)
        Qc = block_diag(*Qc_list)

        x = np.hstack(x_list)
        P = 1000 * np.eye(F.shape[0])
        return {
            'H': H,
            'F': F,
            'L': L,
            'Qc': Qc,
            'x': x,
            'P': P,
        }