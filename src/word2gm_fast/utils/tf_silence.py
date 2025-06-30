"""
TensorFlow Silencing Utilities

This module provides utilities to completely silence TensorFlow's verbose output
and import TensorFlow with minimal noise.
"""

import contextlib
import os
import sys


def setup_tf_silence(deterministic=False, force_cpu=False):
    """
    Set up all TensorFlow environment variables for maximum silencing.
    
    Parameters
    ----------
    deterministic : bool, optional
        If True, enable deterministic operations (requires seeds for random ops).
        Default False to avoid breaking existing random operations.
    force_cpu : bool, optional
        If True, force CPU-only mode by setting CUDA_VISIBLE_DEVICES=-1.
        Default False to allow GPU usage.
    """
    # Core TensorFlow silencing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only FATAL errors
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
    
    # Advanced silencing
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CPP_VMODULE'] = 'computation_placer=0,cuda_dnn=0,cuda_blas=0'
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''


@contextlib.contextmanager
def silence_stderr():
    """Completely redirect stderr to /dev/null (nuclear option)."""
    old_stderr = os.dup(2)  # Save stderr file descriptor
    try:
        # Redirect stderr to /dev/null
        with open('/dev/null', 'w') as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


@contextlib.contextmanager
def log_tf_to_file(file='tf.log'):
    """Redirect TensorFlow messages to a log file."""
    with open(file, 'a') as f:
        old = sys.stderr
        sys.stderr = f
        try:
            yield
        finally:
            sys.stderr = old


def configure_tf_gpu_memory():
    """
    Configure TensorFlow GPU memory to grow dynamically instead of allocating all at once.
    This prevents CUDA out-of-memory errors on large GPUs like A100s.
    Should be called after importing TensorFlow but before creating any operations.
    """
    import tensorflow as tf
    
    # Get list of physical GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(f"Warning: Could not set GPU memory growth: {e}")


def import_tensorflow_silently(deterministic=False, force_cpu=False, gpu_memory_growth=True):
    """
    Import TensorFlow with complete silencing of all messages.
    
    Parameters
    ----------
    deterministic : bool, optional
        If True, enable deterministic operations. Default False.
    force_cpu : bool, optional
        If True, force CPU-only mode. Default False to allow GPU usage.
    gpu_memory_growth : bool, optional
        If True, configure GPU memory to grow dynamically. Default True.
        
    Returns
    -------
    tf
        TensorFlow module
    """
    setup_tf_silence(deterministic=deterministic, force_cpu=force_cpu)


    with silence_stderr():
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.config.experimental.enable_op_determinism()

        # Configure GPU memory growth if requested and not forcing CPU
        if gpu_memory_growth and not force_cpu:
            configure_tf_gpu_memory()

        # (Mixed precision forcibly disabled)

        # Set global random seed for determinism if requested
        if deterministic:
            import random
            import numpy as np
            seed = 1  # Default seed; could be parameterized
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

    return tf


def get_silence_status():
    """Get a summary of current TensorFlow silencing configuration."""
    vars_to_check = [
        'TF_CPP_MIN_LOG_LEVEL', 'TF_ENABLE_ONEDNN_OPTS', 'CUDA_VISIBLE_DEVICES',
        'TF_XLA_FLAGS', 'TF_CPP_VMODULE', 'GRPC_VERBOSITY'
    ]
    
    status = {}
    for var in vars_to_check:
        status[var] = os.environ.get(var, 'NOT SET')
    
    return status


def get_silence_env(deterministic=False, force_cpu=False):
    """
    Get environment variables for TensorFlow silencing in subprocesses.
    
    Parameters
    ----------
    deterministic : bool, optional
        If True, include deterministic operations. Default False.
    force_cpu : bool, optional
        If True, force CPU-only mode. Default False to allow GPU usage.
    """
    env = {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',
        'AUTOGRAPH_VERBOSITY': '0',
        'TF_CPP_VMODULE': 'computation_placer=0,cuda_dnn=0,cuda_blas=0',
        'GRPC_VERBOSITY': 'ERROR',
        'GRPC_TRACE': ''
    }
    
    if force_cpu:
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    
    if deterministic:
        env['TF_DETERMINISTIC_OPS'] = '1'
    
    return env


def run_silent_subprocess(cmd, deterministic=False, force_cpu=False, **kwargs):
    """
    Run a subprocess with TensorFlow silencing environment variables.
    
    Parameters
    ----------
    cmd : list
        Command to run as a list.
    deterministic : bool, optional
        If True, enable deterministic operations. Default False.
    force_cpu : bool, optional
        If True, force CPU-only mode. Default False to allow GPU usage.
    **kwargs
        Additional arguments passed to subprocess.run.
    """
    import subprocess
    
    # Get current environment and update with silencing vars
    env = os.environ.copy()
    env.update(get_silence_env(deterministic=deterministic, force_cpu=force_cpu))
    
    # Set env in kwargs if not already provided
    if 'env' not in kwargs:
        kwargs['env'] = env
    
    result = subprocess.run(cmd, **kwargs)
    
    # Filter out known TensorFlow noise from stderr
    if hasattr(result, 'stderr') and result.stderr:
        tf_noise_patterns = [
            'WARNING: All log messages before absl::InitializeLog',
            'Unable to register cuDNN factory',
            'Unable to register cuBLAS factory', 
            'Unable to register cuFFT factory',
            'computation placer already registered',
            'cuda_dnn.cc',
            'cuda_blas.cc',
            'cuda_fft.cc',
            'computation_placer.cc'
        ]
        
        # Filter stderr lines
        stderr_lines = result.stderr.split('\n')
        filtered_lines = []
        
        for line in stderr_lines:
            # Skip lines containing TensorFlow noise patterns
            if not any(pattern in line for pattern in tf_noise_patterns):
                filtered_lines.append(line)
        
        # Only keep non-empty filtered stderr
        filtered_stderr = '\n'.join(filtered_lines).strip()
        if not filtered_stderr:
            filtered_stderr = None
            
        # Create new result with filtered stderr
        result = subprocess.CompletedProcess(
            args=result.args,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=filtered_stderr
        )
    
    return result
