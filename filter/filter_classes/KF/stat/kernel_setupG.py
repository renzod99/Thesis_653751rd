import numpy as np
from typing import List, Type, Dict, Tuple
from collections import defaultdict
from gpss.kernels.KernelABC import KernelComponent
from gpss.kernel_arithmetics import AdditiveKernel

from gpss.kernel_arithmetics import AdditiveKernel, MultiplicativeKernel, MultiplicativeKernelPeriodic
from gpss.kernels import constant
from gpss.kernels import linear
from gpss.kernels import brownian_motion
from gpss.kernels import matern
from gpss.kernels import exponential
from gpss.kernels import sqe
from gpss.kernels import ratquad
from gpss.kernels import periodic
from gpss.kernels import white_noise

class KernelDescriptor:
    def __init__(self, kernel_class, params=None):
        self.kernel_class = kernel_class
        self.params = kernel_class.get_hyperparameter_keys()

    def __repr__(self):
        return f"KernelDescriptor({self.kernel_class.__name__}, {self.params})"


class CompositeKernelDescriptor:
    def __init__(self, op, children):
        self.op = op
        self.children = children

    def __repr__(self):
        return f"CompositeKernelDescriptor({self.op}, {self.children})"

def process_kernel_string(kernel_str):
    """
    Uses the KernelDescriptor and CompositeKernelDescriptor class to define the kernel and extracts its metadata;hyperparameters
    """
    allowed_kernels = {
        'constant': lambda *args, **kwargs: KernelDescriptor(constant.ConstantKernel, kwargs),
        'linear': lambda *args, **kwargs: KernelDescriptor(linear.LinearKernel, kwargs),
        'brownian_motion': lambda *args, **kwargs: KernelDescriptor(brownian_motion.BrownianMotion, kwargs),
        'brownian_motion_I': lambda *args, **kwargs: KernelDescriptor(brownian_motion.BrownianMotionIntegrated, kwargs),
        'matern12': lambda *args, **kwargs: KernelDescriptor(matern.MaternKernel12, kwargs),
        'matern32': lambda *args, **kwargs: KernelDescriptor(matern.MaternKernel32, kwargs),
        'matern52': lambda *args, **kwargs: KernelDescriptor(matern.MaternKernel52, kwargs),
        'exponential': lambda *args, **kwargs: KernelDescriptor(exponential.ExponentialKernel, kwargs),
        'sqe': lambda *args, **kwargs: KernelDescriptor(sqe.SQEKernel, kwargs),
        'ratquad': lambda *args, **kwargs: KernelDescriptor(ratquad.RQKernel, kwargs),
        'periodic': lambda *args, **kwargs: KernelDescriptor(periodic.PeriodicKernel, kwargs),
        'periodic_lists': lambda *args, **kwargs: KernelDescriptor(periodic.PeriodicKernelLists, kwargs),
        'periodic_stochastic': lambda *args, **kwargs: KernelDescriptor(periodic.PeriodicKernelStochastic, kwargs),
        'periodic_stochastic_lists': lambda *args, **kwargs: KernelDescriptor(periodic.PeriodicKernelStochasticLists, kwargs),
        'white_noise': lambda *args, **kwargs: KernelDescriptor(white_noise.WhiteNoiseKernel, kwargs),
        # Combining functions produce composite descriptors
        'Add': lambda *kernels: CompositeKernelDescriptor("Add", list(kernels)),
        'Mult': lambda base, multiplier: CompositeKernelDescriptor("Mult", [base, multiplier]),
        'MultPeriodic': lambda base, period: CompositeKernelDescriptor("MultPeriodic", [base, period]),
    }
    
    descriptor = eval(kernel_str, {"__builtins__": None}, allowed_kernels)
    metadata = extract_descriptor_metadata(descriptor)
    return descriptor, metadata

def extract_descriptor_metadata(descriptor) -> Dict[Tuple[str, int], List[str]]:
    """
    Recursively traverse the kernel descriptor tree and extract hyperparameter keys.
    Each leaf (KernelDescriptor) is assigned a unique instance number.
    """
    kernel_count = defaultdict(int)
    metadata = {}

    def recursive_extract(desc):
        if isinstance(desc, KernelDescriptor):
            cls_name = desc.kernel_class.__name__
            kernel_count[cls_name] += 1
            instance_id = kernel_count[cls_name]
            metadata[(cls_name, instance_id)] = desc.params
        elif isinstance(desc, CompositeKernelDescriptor):
            for child in desc.children:
                recursive_extract(child)

    recursive_extract(descriptor)
    metadata[("measurement variance", 1)] = ["R"]
    return metadata

def prep_kernel_metadata_optimization(kernel_metadata):
    """
    converts the hyperparameters in the metadata dictionary to a list in the order of the keys so it is suitable for input
    in scipy optimize
    """
    param_keys = []
    for (kernel_name, instance_id), hyperparams in kernel_metadata.items():
        for param in hyperparams:
            param_keys.append((kernel_name, instance_id, param))
    return param_keys

def reconstruct_kernels(param_keys, params):
    reconstructed_kernels = {}
    for (kernel_name, instance_id, param), value in zip(param_keys, params):
        if (kernel_name, instance_id) not in reconstructed_kernels:
            reconstructed_kernels[(kernel_name, instance_id)] = {}
        reconstructed_kernels[(kernel_name, instance_id)][param] = value
    return reconstructed_kernels

def instantiate_kernels_from_descriptor(descriptor, optimized_kernel_structure):
    """
    Given a kernel descriptor and an optimized parameter dictionary
    recursively instantiates the kernel
    """
    kernel_count = defaultdict(int)
    
    def _instantiate(desc):
        if isinstance(desc, KernelDescriptor):
            cls_name = desc.kernel_class.__name__
            kernel_count[cls_name] += 1
            instance_id = kernel_count[cls_name]
            kernel_key = (cls_name, instance_id)
            
            if kernel_key in optimized_kernel_structure:
                hyperparams = optimized_kernel_structure[kernel_key]
                # Convert all hyperparameter values to floats.
                hyperparams = {k: float(v) for k, v in hyperparams.items()}
                instance = desc.kernel_class(**hyperparams)
            else:
                instance = desc.kernel_class()
            return instance

        elif isinstance(desc, CompositeKernelDescriptor):
            instantiated_children = [_instantiate(child) for child in desc.children]
            if desc.op == "Add":
                return AdditiveKernel(instantiated_children)
            elif desc.op == "Mult":
                if len(instantiated_children) != 2:
                    raise ValueError("Mult operation requires exactly two kernels")
                return MultiplicativeKernel(instantiated_children[0], instantiated_children[1])
            elif desc.op == "MultPeriodic":
                if len(instantiated_children) != 2:
                    raise ValueError("MultPeriodic operation requires exactly two kernels")
                return MultiplicativeKernelPeriodic(instantiated_children[0], instantiated_children[1])
            else:
                raise ValueError(f"Unknown composite operation: {desc.op}")

        else:
            raise TypeError("Unknown descriptor type")
    return _instantiate(descriptor)

def construct_kernel_ss_mats(descriptor, hyperparams):
    """
    practical function that from a kernel descriptor and hyperparameters instantiates the state space matrices
    """
    kernel = instantiate_kernels_from_descriptor(descriptor, hyperparams)
    ss_mats = kernel.initialize()
    ss_mats["R"] = hyperparams['measurement variance', 1]["R"]
    return ss_mats
