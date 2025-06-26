"""
Notebook Setup Utility

Standardized setup for Word2GM notebooks including path configuration,
environment setup, and dependency imports.
"""

import os
import sys
import warnings
from pathlib import Path
from IPython.display import display, Markdown


def setup_notebook_environment(
    project_root: str = '/scratch/edk202/word2gm-fast',
    force_cpu: bool = False,
    gpu_memory_growth: bool = True
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

    setup_lines = [
        f"Project root: {project_root}",
        f"TensorFlow version: {tf.__version__}",
        f"Device mode: {setup_info['device_mode']}"
    ]
    display(Markdown(f"<pre>{'\n'.join(setup_lines)}</pre>"))

    return setup_info


def run_silent_subprocess(cmd, **kwargs):
    """
    Run a subprocess and return the completed process object.
    By default, suppresses output unless capture_output=True is passed.
    """
    import subprocess
    # Remove 'deterministic' if present (not a valid subprocess arg)
    kwargs.pop('deterministic', None)
    return subprocess.run(cmd, **kwargs)


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
        gpu_memory_growth=False
    )
    
    # Import data preprocessing modules
    from word2gm_fast.dataprep.pipeline import batch_prepare_training_data
    from word2gm_fast.utils.resource_summary import print_resource_summary
    
    setup_info['batch_prepare_training_data'] = batch_prepare_training_data
    setup_info['print_resource_summary'] = print_resource_summary
    setup_info['run_silent_subprocess'] = run_silent_subprocess
    # Only print a single concise confirmation
    display(Markdown("<pre>Data preprocessing environment ready!</pre>"))
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
    setup_info['run_silent_subprocess'] = run_silent_subprocess
    # Only print a single concise confirmation
    display(Markdown("<pre>Training environment ready!</pre>"))
    return setup_info


def setup_testing_notebook(
    project_root: str = '/scratch/edk202/word2gm-fast'
):
    """
    Specialized setup for testing notebooks.
    
    Configures GPU mode and imports all test functions from each test module for programmatic access.
    
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
    )
    
    # Import all test functions from each test module
    import tests.test_corpus_to_dataset as test_corpus_to_dataset_mod
    import tests.test_dataset_to_triplets as test_dataset_to_triplets_mod
    import tests.test_index_vocab as test_index_vocab_mod
    import tests.test_tfrecord_io as test_tfrecord_io_mod
    import tests.test_word2gm_model as test_word2gm_model_mod
    from word2gm_fast.utils.resource_summary import print_resource_summary

    # Add all test functions from each module to setup_info
    for mod, key in [
        (test_corpus_to_dataset_mod, 'test_corpus_to_dataset'),
        (test_dataset_to_triplets_mod, 'test_dataset_to_triplets'),
        (test_index_vocab_mod, 'test_index_vocab'),
        (test_tfrecord_io_mod, 'test_tfrecord_io'),
        (test_word2gm_model_mod, 'test_word2gm_model'),
    ]:
        for attr in dir(mod):
            if attr.startswith('test_'):
                setup_info[f'{key}.{attr}'] = getattr(mod, attr)

    setup_info['print_resource_summary'] = print_resource_summary
    setup_info['run_silent_subprocess'] = run_silent_subprocess
    display(Markdown("<pre>Testing environment ready!</pre>"))
    return setup_info


def enable_autoreload():
    """Enable IPython autoreload for development."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython is None:
            display(Markdown("<pre>Not in IPython environment - autoreload not available</pre>"))
            return False
        
        # Load autoreload extension and enable mode 2
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        display(Markdown("<pre>Autoreload enabled</pre>"))
        return True
        
    except ImportError:
        display(Markdown("<pre>IPython not available - autoreload not supported</pre>"))
        return False
    except Exception as e:
        display(Markdown(f"<pre>Could not configure autoreload: {e}</pre>"))
        return False
