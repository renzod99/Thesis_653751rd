import logging
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from filter.filter_classes.KF.stat import kernel_setupG
from filter.filter_classes.TFilter import TFilter, TFilterInterpreter, kernelsetupT
from filter.filter_classes.KF.stat import KF


logger = logging.getLogger(__name__)

def mml(train, test, kernel_str, bounds_file_path:str, number_of_init:int=1):
    descriptor, kernel_metadata = kernel_setupG.process_kernel_string(kernel_str)
    keys = kernel_setupG.prep_kernel_metadata_optimization(kernel_metadata)

    def objective(params):
        reconstructed_kernels = kernel_setupG.reconstruct_kernels(keys, params)
        # for key, item in reconstructed_kernels.items():
        #     print(f"{key}: {item}")
        ss_mats = kernel_setupG.construct_kernel_ss_mats(descriptor, reconstructed_kernels)
        kf = KF.KalmanFilter(ss_mats)
        kf.run_filter(train)        
        return kf.negative_log_likelihood
    
    best_result = None
    best_nnl = np.inf
    for i in range(number_of_init):
        bounds = get_custom_bounds(bounds_file_path)
        init = initialize_random_vals(bounds)
        result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
        logging.info(f"minimization success:{result.success}, NLL: {result.fun}")
        if result.fun < best_nnl and result.success:
            best_result = result
    optimized_hyperparams = kernel_setupG.reconstruct_kernels(keys, best_result.x)
    return optimized_hyperparams, result

def mml_t(train, test, kernel_str, bounds_file_path:str, number_of_init:int=1):
    descriptor, kernel_metadata = kernelsetupT.process_kernel_string(kernel_str)
    keys = kernelsetupT.prep_kernel_metadata_optimization(kernel_metadata)
    def objective(params):
        reconstructed_kernels = kernelsetupT.reconstruct_kernels(keys, params)
        # for key, item in reconstructed_kernels.items():
        #     print(f"{key}: {item}")
        ss_mats = kernelsetupT.construct_kernel_ss_mats(descriptor, reconstructed_kernels)
        kf = TFilter.StudentTFilter(ss_mats)
        kf.run_filter(train)
        return kf.negative_log_likelihood
    
    best_result = None
    best_nnl = np.inf
    for i in range(number_of_init):
        bounds = get_custom_bounds(bounds_file_path)
        init = initialize_random_vals(bounds)
        result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
        logging.info(f"minimization status:{result.success}, NLL: {result.fun}")
        if result.success and result.fun < best_nnl:
            best_result = result
            best_nnl = result.fun
    
    logging.info(f"Final result minimization, NLL: {best_nnl}")
    optimized_hyperparams = kernelsetupT.reconstruct_kernels(keys, best_result.x)
    return optimized_hyperparams, result

def get_custom_bounds(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    if "bounds" not in config:
        raise KeyError("Config file must contain a 'bounds' section.")
    
    bounds_dict = config["bounds"]
    bounds = list(bounds_dict.values())
    return bounds

def initialize_random_vals(bounds):
    initial_values = []
    for bound in bounds:
        lb, ub = bound
        lb, ub = float(lb), float(ub)
        if ub == np.inf:
            ub = 10
        initial_values.append(np.random.uniform(lb, ub))
    return initial_values