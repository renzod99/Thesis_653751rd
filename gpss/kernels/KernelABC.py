from abc import ABC, abstractmethod

class KernelComponent(ABC):
    @classmethod
    def get_hyperparameter_keys(cls):
        """
        Returns the hyperparameter keys for the kernel.
        This method ensures all subclasses define their hyperparameters.
        """
        return cls.hyperparameter_keys
    
    @abstractmethod
    def initialize(self):
        """
        Initialize the state-space representation of the kernel.
        Should return a dictionary with keys: H, F, L, Qc, x, P
        """
        pass
    
    def characteristic_coefficients(self):
        """
        calculates the coefficients for the state transition matrix bottom row
        Should return an array of negative coefficients and in some cases derivatives with respect to a hyperparameter
        """
        pass
    
    def calculate_Qc(self):
        """
        calculates the spectral density of the respective kernel
        Should return Qc, which can be either scalar or matrix
        """
        pass