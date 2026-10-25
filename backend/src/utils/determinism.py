"""
Determinism utilities for reproducible experiments.

Sets all random seeds and TensorFlow flags to ensure reproducible results.
"""
import os
import random

import numpy as np
import tensorflow as tf


def set_full_determinism(seed=42):
    """
    Set full determinism for reproducible experiments.
    
    This function sets all random seeds and TensorFlow flags to minimize
    non-determinism from GPU operations, threading, and hash randomization.
    
    Args:
        seed: Random seed value (default: 42)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Force single-thread to eliminate race conditions
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def force_cpu():
    """
    Force TensorFlow to use CPU instead of GPU.
    
    This is slower but more deterministic for research comparisons.
    Call this before set_full_determinism() if GPU non-determinism persists.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
