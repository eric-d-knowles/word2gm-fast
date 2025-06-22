"""
TensorFlow Silencing Utilities

This module provides utilities to completely silence TensorFlow's verbose output,
including C++ library messages that bypass Python logging systems.
"""

import contextlib
import os
import sys


def setup_tf_silence():
    """Set up all TensorFlow environment variables for maximum silencing."""
    # Core TensorFlow silencing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only FATAL errors
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
    
    # Advanced silencing
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
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


def import_tensorflow_silently():
    """Import TensorFlow with complete silencing of all messages."""
    setup_tf_silence()
    
    with silence_stderr():
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.config.experimental.enable_op_determinism()
    
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


def get_silence_env():
    """Get environment variables for TensorFlow silencing in subprocesses."""
    return {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'CUDA_VISIBLE_DEVICES': '-1',
        'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices=false',
        'AUTOGRAPH_VERBOSITY': '0',
        'TF_DETERMINISTIC_OPS': '1',
        'TF_CPP_VMODULE': 'computation_placer=0,cuda_dnn=0,cuda_blas=0',
        'GRPC_VERBOSITY': 'ERROR',
        'GRPC_TRACE': ''
    }


def run_silent_subprocess(cmd, **kwargs):
    """Run a subprocess with TensorFlow silencing environment variables."""
    import subprocess
    
    # Get current environment and update with silencing vars
    env = os.environ.copy()
    env.update(get_silence_env())
    
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
