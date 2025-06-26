"""
Notebook Setup Utility

Standardized setup for Word2GM notebooks including path configuration,
environment setup, and dependency imports.
"""

import os
import sys
import warnings
from pathlib import Path


def setup_notebook_environment(
    project_root: str = '/scratch/edk202/word2gm-fast',
    force_cpu: bool = False,
    gpu_memory_growth: bool = True,
    verbose: bool = True
):
    """
    Set up the notebook environment for Word2GM development.
    
    Parameters
    ----------
    project_root : str
        Path to the project root directory
    force_cpu : bool
        Whether to force CPU-only mode (useful for data preprocessing)
    gpu_memory_growth : bool
        Whether to enable GPU memory growth
    verbose : bool
        Whether to print setup information
        
    Returns
    -------
    dict
        Dictionary containing imported modules and setup information
    """
    setup_info = {}
    
    # Setup project path (only if not already configured)
    project_root = Path(project_root)
    src_path = project_root / 'src'
    
    # Only change directory if we're not already there
    if Path.cwd() != project_root:
        os.chdir(project_root)
    
    # Only add to sys.path if not already present
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    setup_info['project_root'] = project_root
    setup_info['src_path'] = src_path
    
    # Configure GPU/CPU mode
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        setup_info['device_mode'] = 'CPU-only'
    else:
        setup_info['device_mode'] = 'GPU-enabled'
    
    # Import TensorFlow with proper configuration
    from word2gm_fast.utils.tf_silence import import_tensorflow_silently
    tf = import_tensorflow_silently(
        force_cpu=force_cpu, 
        gpu_memory_growth=gpu_memory_growth
    )
    setup_info['tensorflow'] = tf
    
    # Import common dependencies
    import numpy as np
    import pandas as pd
    import time
    import psutil
    
    setup_info['numpy'] = np
    setup_info['pandas'] = pd
    setup_info['time'] = time
    setup_info['psutil'] = psutil
    
    if verbose:
        print(f"Project root: {project_root}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Device mode: {setup_info['device_mode']}")
        if force_cpu:
            print("   (Optimal for data preprocessing + multiprocessing)")
        print("Setup complete; all modules loaded successfully")
    
    return setup_info


def setup_data_preprocessing_notebook(
    project_root: str = '/scratch/edk202/word2gm-fast'
):
    """
    Specialized setup for data preprocessing notebooks.
    
    Configures CPU-only mode and imports data processing modules.
    
    Parameters
    ----------
    project_root : str
        Path to the project root directory
    """
    # Basic setup with CPU-only mode
    setup_info = setup_notebook_environment(
        project_root=project_root,
        force_cpu=True,
        gpu_memory_growth=False,
        verbose=True
    )
    
    # Import data preprocessing modules
    from word2gm_fast.dataprep.pipeline import batch_prepare_training_data
    from word2gm_fast.utils.resource_summary import print_resource_summary
    
    setup_info['batch_prepare_training_data'] = batch_prepare_training_data
    setup_info['print_resource_summary'] = print_resource_summary
    
    return setup_info


def setup_training_notebook(project_root: str = '/scratch/edk202/word2gm-fast'):
    """
    Specialized setup for training notebooks.
    
    Configures GPU mode and imports training modules.
    
    Parameters
    ----------
    project_root : str
        Path to the project root directory
    """
    # Basic setup with GPU enabled
    setup_info = setup_notebook_environment(
        project_root=project_root,
        force_cpu=False,
        gpu_memory_growth=True,
        verbose=True
    )
    
    # Import training modules
    from word2gm_fast.models.word2gm_model import Word2GMModel
    from word2gm_fast.models.config import Word2GMConfig
    from word2gm_fast.training.training_utils import train_step
    from word2gm_fast.utils.resource_summary import print_resource_summary
    
    setup_info['Word2GMModel'] = Word2GMModel
    setup_info['Word2GMConfig'] = Word2GMConfig
    setup_info['train_step'] = train_step
    setup_info['print_resource_summary'] = print_resource_summary
    
    return setup_info


def enable_autoreload():
    """Enable IPython autoreload for development."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython is None:
            print("Not in IPython environment - autoreload not available")
            return False
        
        # Load autoreload extension and enable mode 2
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        print("Autoreload enabled")
        return True
        
    except ImportError:
        print("IPython not available - autoreload not supported")
        return False
    except Exception as e:
        print(f"Could not configure autoreload: {e}")
        return False
