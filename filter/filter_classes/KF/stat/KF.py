import logging
import numpy as np

from filter import kf_utility

logger = logging.getLogger(__name__)

class KalmanFilter:
    """
    Kalman filter class based on the 1960 paper by Kalman and Särrkä 2013.
    
    @article{kalman1960new,
    author    = {Kalman, R. E.},
    title     = {A New Approach to Linear Filtering and Prediction Problems},
    journal   = {Transactions of the ASME--Journal of Basic Engineering},
    year      = {1960},
    volume    = {82},
    number    = {1},
    pages     = {35--45},
    publisher = {American Society of Mechanical Engineers}
    }

    @book{sarkka2013bayesian,
    author    = {Särkkä, Simo},
    title     = {Bayesian Filtering and Smoothing},
    year      = {2013},
    publisher = {Cambridge University Press},
    address   = {Cambridge, UK},
    series    = {IMS Textbooks},
    volume    = {3},
    url       = {http://www.cambridge.org/sarkka}
    }
    """
    def __init__(self, ss_matrices):
        self.H = ss_matrices["H"]
        self.F = ss_matrices["F"]
        self.L = ss_matrices["L"]
        self.Qc = ss_matrices["Qc"]
        self.x = ss_matrices["x"] 
        self.P = ss_matrices["P"]
        self.R = ss_matrices["R"]

        self.I = np.eye(ss_matrices["H"].shape[0])        
        self.negative_log_likelihood = 0

        self.filtered_estimates = []
        self.forecast_estimates = []
        self.rolling_forecasts = []

        kf_utility.is_stable_continuous(self.F)

    def get_discretized_system(self, dt):
        Fd, Qd = kf_utility.discretize_lti_system(self.F, self.L, self.Qc, dt)
        return Fd, Qd
    
    def predict(self, Fd, Qd):
        """
        Kalman prediction. 
        Propagates the latent states through discretized state tranistion matrix Fd.
        Propagates uncertainty through Fd and discretized process noise matrix Qd.
        """
        x_prior = Fd @ self.x
        P_prior = Fd @ self.P @ Fd.T + Qd
        P_prior = kf_utility.ensure_psd(P_prior)
        return x_prior, P_prior
    
    def update(self, z, Fd, Qd):
        """
        Kalman update.
        Uses the kalman equations to update the latent states and uncertainty.
        """
        x_prior, P_prior = self.predict(Fd, Qd)
        y = z - np.dot(self.H, x_prior)
        S = np.dot(self.H, np.dot(P_prior, self.H.T)) + self.R
        if S <= 0:
            raise ValueError(f"S cannot be negative: {S}")
        
        self.negative_log_likelihood += 0.5 * (np.log(2 * np.pi) + np.log(S) + (y**2) / S)

        K = np.dot(P_prior, self.H) / S
        I_KH = self.I - np.outer(K, self.H)
        
        self.x = x_prior + np.dot(K, y)
        self.P = I_KH @ P_prior @ I_KH.T + np.outer(K, K) * self.R
        self.filtered_estimates.append((self.x, self.P))

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
    

