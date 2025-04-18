import logging
import numpy as np
from scipy.linalg import block_diag
from scipy.special import gammaln

from filter import kf_utility

logger = logging.getLogger(__name__)
class StudentTFilter:
    """
    This class is based of the paper of Solin (2016).
    @phdthesis{Solin2016,
    author       = {A. Solin},
    title        = {Stochastic Differential Equation Methods for Spatio-Temporal Gaussian Process Regression},
    school       = {Department of Neuroscience and Biomedical Engineering / Department of Computer Science, Aalto University},
    year         = {2016},
    series       = {Aalto University publication series DOCTORAL DISSERTATIONS},
    number       = {216/2016},
    address      = {Espoo, Finland}
    }
    """
    def __init__(self, ss_matrices):
        self.H = ss_matrices["H"]
        self.F = ss_matrices["F"]
        self.L = ss_matrices["L"]
        self.Qc = ss_matrices["Qc"]
        self.x = ss_matrices["x"] 
        self.P = ss_matrices["P"]
        self.sigma_n = ss_matrices["R"]
        self.gamma = 1
        self.nu = ss_matrices["nu"]
        self.negative_log_likelihood = 0

        self.Htrain = np.hstack([self.H, np.array([1])])
        self.Htest = np.hstack([self.H, np.array([0])])
        self.F = block_diag(self.F, np.array([[-1e2]]))
        self.L = block_diag(self.L, np.zeros((1, 1)))
        self.Qc = block_diag(self.Qc, np.array([[self.sigma_n**2]]))
        self.x = np.hstack([self.x, np.zeros(1)])
        self.P = block_diag(self.P, np.array([[self.sigma_n**2]]))
        
        self.filtered_estimates=[]
        self.filtered_values=[]
        self.forecast_estimates = []
        self.rolling_forecasts = []

        kf_utility.is_stable_continuous(self.F)
    
    def get_discretized_system(self, dt):
        Fd, Qd = kf_utility.discretize_lti_system(self.F, self.L, self.Qc, dt)
        return Fd, Qd
    
    def predict(self, Fd, Qd):
        x_prior = np.dot(Fd, self.x)
        P_prior = np.dot(Fd, np.dot(self.P, Fd.T)) + self.gamma * Qd
        P_prior = kf_utility.ensure_psd(P_prior)
        return x_prior, P_prior
    
    def update(self, z, Fd, Qd):
        x_prior, P_prior = self.predict(Fd, Qd)

        yk = z - np.dot(self.Htrain, x_prior)
        S = np.dot(self.Htrain, np.dot(P_prior, self.Htrain.T))
        if S <= 0:
           raise ValueError(f"S (Innovation Covariance) is non-positive: S={S}")

        K = np.dot(P_prior, self.Htrain) / S
        
        nu_prev = self.nu - 1 if self.nu > 3 else self.nu
        self.gamma = (self.gamma / (self.nu - 2)) * (nu_prev - 2 + ((yk**2) / S))
        gamma_minus1 = self.filtered_estimates[-1][2] if self.filtered_estimates else 1
        self.x = x_prior + np.dot(K, yk)
        self.P = (self.gamma / gamma_minus1 ) * (P_prior - S * np.outer(K, K))
        self.P = kf_utility.ensure_psd(self.P)
        
        # self.nu += 1
        # nu_prev = self.nu - 1
        term1 = 0.5 * np.log((self.nu - 2) * np.pi)
        term2 = 0.5 * np.log(S)
        term3 = gammaln(nu_prev / 2) - gammaln(self.nu / 2)
        term4 = 0.5 * np.log((nu_prev - 2) / (self.nu - 2))
        term5 = (self.nu / 2) * np.log(1 + (yk**2) / ((nu_prev - 2) * S))
        self.negative_log_likelihood += (term1 + term2 + term3 + term4 + term5)
        
        self.filtered_estimates.append((self.x.copy(), self.P.copy(), self.gamma.copy()))

    def run_filter(self, filter_data):
        for idx, z in enumerate(filter_data["quantity"]):
            dt = filter_data.iloc[idx, filter_data.columns.get_loc('time_diff')]
            self.update(z, *self.get_discretized_system(dt))
    
    def run_forecast(self, horizon):
        for i in range(len(horizon)):
            dt = 1
            self.x, self.P = self.predict(*self.get_discretized_system(dt))
            self.forecast_estimates.append((self.x, self.P))

        self.x = self.filtered_estimates[-1][0]
        self.P = self.filtered_estimates[-1][1]
    
    def run_rolling_forecast(self, forecast_data, horizon):
        for idx, z in enumerate(forecast_data["quantity"]):
            current_forecast = []
            for h in range(1, horizon + 1):
                x_temp, p_temp = self.predict(*self.get_discretized_system(h))
                current_forecast.append((x_temp, p_temp))
            self.rolling_forecasts.append(current_forecast)
            dt_update = forecast_data.iloc[idx, forecast_data.columns.get_loc('time_diff')]          
            self.update(z, *self.get_discretized_system(dt_update))