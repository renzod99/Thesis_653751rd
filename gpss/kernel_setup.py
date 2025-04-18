import numpy as np
from typing import List, Type, Dict, Tuple
from collections import defaultdict
from gpss.kernels.KernelABC import KernelComponent
from gpss.kernel_arithmetics import AdditiveKernel


def return_kernel_metadata(kernel_structure: List[Type[KernelComponent]]):
    kernel_count = defaultdict(int)
    kernel_metadata = defaultdict(list)

    for kernel_cls in kernel_structure:
        cls_name = kernel_cls.__name__
        kernel_count[cls_name] += 1
        instance_id = kernel_count[cls_name]

        keys = kernel_cls.get_hyperparameter_keys() or []
        kernel_metadata[(cls_name, instance_id)].extend(keys)
    kernel_metadata[("measurement variance", 1)].append("R")
    return kernel_metadata

def prep_kernel_metadata_optimization(kernel_metadata):
    param_keys = []
    for (kernel_name, instance_id), hyperparams in kernel_metadata.items():
        for param in hyperparams:
            param_keys.append((kernel_name, instance_id, param))  # Store kernel mapping
    return param_keys

def reconstruct_kernels(param_keys, params):
    reconstructed_kernels = {}
    for (kernel_name, instance_id, param), value in zip(param_keys, params):
        if (kernel_name, instance_id) not in reconstructed_kernels:
            reconstructed_kernels[(kernel_name, instance_id)] = {}
        reconstructed_kernels[(kernel_name, instance_id)][param] = value
    return reconstructed_kernels

def instantiate_kernels(kernel_structure: List[Type[KernelComponent]], optimized_kernel_structure: Dict):
    instantiated_kernels = []  # List to store instantiated kernel objects
    kernel_count = defaultdict(int)  # Track occurrences of each kernel type

    for kernel_cls in kernel_structure:
        cls_name = kernel_cls.__name__
        kernel_count[cls_name] += 1
        instance_id = kernel_count[cls_name]
        kernel_key = (cls_name, instance_id)

        if kernel_key in optimized_kernel_structure:
            hyperparams = optimized_kernel_structure[kernel_key]
            hyperparams = {key: float(value) for key, value in hyperparams.items()}
            kernel_instance = kernel_cls(**hyperparams)  # Instantiate with parameters
        else:
            kernel_instance = kernel_cls()  # Instantiate without parameters

        instantiated_kernels.append(kernel_instance)
    return instantiated_kernels

def construct_additive_kernel(kernel_structure, hyperparams):
    instantiated_kernels = instantiate_kernels(kernel_structure, hyperparams)
    ss_matrices = AdditiveKernel(instantiated_kernels).initialize()
    ss_matrices["R"] = hyperparams['measurement variance', 1]["R"]
    return ss_matrices