import numpy as np
import pandas as pd
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import norm

from filter import kf_utility
from filter.filter_classes.KF.stat import KF

class KFInterpreter:
    def __init__(self, kf: KF.KalmanFilter, transformation:float=None):
        self.kf = kf
        self.scaling_factor = transformation

    def process_filtered_estimates(self):
        filtered_means, filtered_std = [], []
        for x, P in self.kf.filtered_estimates:
            filtered_means.append(np.dot(self.kf.H, x))
            filtered_std.append(np.sqrt(np.dot(self.kf.H, np.dot(P, self.kf.H.T))))
        filtered_means, filtered_std = np.array(filtered_means), np.array(filtered_std)
        return filtered_means, filtered_std
    
    def process_forecast(self):
        forecast_means = []
        forecast_std = []
        for x, p in self.kf.forecast_estimates:
            forecast_means.append(np.dot(self.kf.H, x))
            forecast_std.append(np.sqrt(np.dot(self.kf.H, np.dot(p, self.kf.H.T))))
        forecast_means, forecast_std = np.array(forecast_means), np.array(forecast_std)
        return forecast_means, forecast_std
    
    def process_rolling_forecasts(self):
        horizon = len(self.kf.rolling_forecasts[0])
        forecast_means, forecast_std = [[] for _ in range(horizon)], [[] for _ in range(horizon)]
        for window in self.kf.rolling_forecasts:
            for h in range(horizon):
                mean = np.dot(self.kf.H, window[h][0])
                std = np.sqrt(np.dot(self.kf.H, np.dot(window[h][1], self.kf.H.T)))
                
                forecast_means[h].append(mean)
                forecast_std[h].append(std)
        forecast_means = [np.array(horizon) for horizon in forecast_means]
        forecast_std = [np.array(horizon) for horizon in forecast_std]
        return (forecast_means, forecast_std)
    
    def return_forecast_errors(self, test, return_full:bool=True):
        forecast_means, forecast_std = self.process_forecast()
        if self.scaling_factor:
            forecast_means, forecast_std = self.reverse_transform(forecast_means, forecast_std)
        individual, summary = kf_utility.calculate_forecast_errors(forecast_means, test["quantity"].to_numpy(), return_full=return_full)
        return individual, summary
    
    def return_forecast_errors_rolling(self, test_data, return_full:bool=False):
        forecast_means, forecast_std = self.process_rolling_forecasts()

        if self.scaling_factor:
            forecast_means, forecast_std = self.reverse_transform(forecast_means, forecast_std)

        forecast_errors = []
        for h, means in enumerate(forecast_means):
            actual_shifted = test_data.iloc[h:, test_data.columns.get_loc('quantity')].to_numpy()
            aligned_preds = means[:len(actual_shifted)]
            
            individual, summary = kf_utility.calculate_forecast_errors(aligned_preds, actual_shifted, return_full=return_full)
            forecast_errors.append((individual, summary))
        return forecast_errors
    
    def plot_results(self, train, test):
        filtered_means, filtered_std = self.process_filtered_estimates()
        forecast_means, forecast_std = self.process_forecast()
        
        full_time, time_train, time_test = prep_plot_x_axis(train, test)
        full_data, lower_bounds, upper_bounds = prep_plot_y_axis(train, test, forecast_means, forecast_std, 0.05)

        plt.figure(figsize=(12, 6))
        plt.plot(full_time, full_data['quantity'], '--', label='Observations', color="black", linewidth=2)
        plt.plot(time_train, filtered_means[time_train], '-', label='Filtered Estimate', color='yellow', linewidth=2)
        plt.fill_between(range(filtered_means.shape[0]),
                    filtered_means - filtered_std,
                    filtered_means + filtered_std,
                    color='red', alpha=0.2, label='±1 Std Dev')
        
        plt.plot(time_test, forecast_means, '--', label='Forecasted Estimate', color='blue')
        plt.fill_between(time_test, lower_bounds, upper_bounds,
                        color='blue', alpha=0.2, label='Forecast ± Std Dev')
        plt.title('Kalman filter results')
        plt.xlabel('Time Step')
        plt.ylabel('Measurement')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_results_rolling(self, train, test):
        filtered_means, filtered_std = self.process_filtered_estimates()
        forecast_means, forecast_std = self.process_rolling_forecasts()

        if self.scaling_factor:
            filtered_means, filtered_std = self.reverse_transform(filtered_means, filtered_std)
            forecast_means, forecast_std = self.reverse_transform(forecast_means, forecast_std)
        
        horizon = len(self.kf.rolling_forecasts[0])
        full_time, time_train, time_test, time_h_window = prep_plot_x_axis(train, test, horizon)
        full_data, lower_bounds, upper_bounds = prep_plot_y_axis(train, test, forecast_means, forecast_std, 0.05)
        forecast_labels, forecast_colors = plot_labels_and_colours_rolling(horizon)

        fig, axes = plt.subplots(horizon, 1, figsize=(12, 12), sharex=True)
        if horizon == 1:
            axes = [axes]
        for h in range(horizon):
            axes[h].plot(full_time, full_data['quantity'], 'o--', label='Observations', color="black", linewidth=2)
            axes[h].plot(time_train, filtered_means[time_train], '-', label='Filtered Estimate', color='yellow', linewidth=2)
            axes[h].plot(time_test, filtered_means[time_test], '-', label='Filtered Estimate after hyperparameter optimization', color='blue', linewidth=2)
            
            axes[h].plot(time_h_window[h], forecast_means[h], 's--', label=forecast_labels[h], color=forecast_colors[h], linewidth=1.5)
            axes[h].fill_between(time_h_window[h],
                                lower_bounds[h], 
                                upper_bounds[h], 
                                color=forecast_colors[h], 
                                label='90 percent confidence interval',
                                alpha=0.3)
            axes[h].set_ylabel('Measurement')
            axes[h].set_title(forecast_labels[h])
            axes[h].legend()
            axes[h].grid(True)
        
        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        plt.show()

    def reverse_transform(self, mean, std):
        if isinstance(mean, list):
            return [v / self.scaling_factor for v in mean], [s / self.scaling_factor for s in std]
        else:
            return mean / self.scaling_factor, std/self.scaling_factor
        
def prep_plot_x_axis(train, test, horizon:int=None):
    n_train = len(train)
    n_test = len(test)
    
    full_time = np.arange(n_train + n_test)
    time_train = np.arange(len(train))
    time_forecast = np.arange(len(train), len(train) + len(test))
    if horizon:
        time_forecast_h = [np.arange(n_train + h, n_train + n_test - h) for h in range(horizon)]
        return full_time, time_train, time_forecast, time_forecast_h
    return full_time, time_train, time_forecast

def prep_plot_y_axis(train, test, forecast_means, forecast_std, alpha):
    full_data = pd.concat([train, test], axis=0, ignore_index=True)

    alpha = alpha
    if isinstance(forecast_means, list) and isinstance(forecast_std, list):
        lower_bounds, upper_bounds = [[] for _ in range(len(forecast_means))], [[] for _ in range(len(forecast_std))]
        for i, horizon in enumerate(forecast_means):
            for mu, sigma in zip(forecast_means, forecast_std[i]):
                lower = norm.ppf(alpha / 2, loc=mu, scale=sigma)
                upper = norm.ppf(1 - alpha / 2, loc=mu, scale=sigma)
                lower_bounds[i].append(lower)
                upper_bounds[i].append(upper)
        
        lower_bounds = [np.array(horizon).squeeze() for horizon in lower_bounds]
        upper_bounds = [np.array(horizon).squeeze() for horizon in upper_bounds]
    else:
        lower_bounds, upper_bounds = [], []
        for mu, sigma in zip(forecast_means, forecast_std):
            lower = norm.ppf(alpha / 2, loc=mu, scale=sigma)
            upper = norm.ppf(1 - alpha / 2, loc=mu, scale=sigma)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
    return full_data, lower_bounds, upper_bounds
    
def plot_labels_and_colours_rolling(horizon):
    forecast_colors = [cm.get_cmap('tab10').colors[i % 10] for i in range(horizon)]
    forecast_labels = [f"{i+1}_step Forecast" for i in range(horizon)]
    return forecast_labels, forecast_colors