"""
GPU Configuration Utilities

This module provides utilities for:
- GPU detection and accessibility checking
- TensorFlow GPU configuration and setup
- Device selection and memory management
- GPU diagnostics and monitoring
- Cluster resource reporting and monitoring
"""

import os
import sys
import subprocess
import psutil
import socket
from typing import Optional, List, Dict, Any


def detect_accessible_gpu() -> Optional[int]:
    """
    Detect accessible GPU using device file approach.
    
    This method works reliably in HPC environments where nvidia-smi
    enumeration may fail but GPU devices are actually accessible.
    
    Returns
    -------
    int or None
        GPU ID of the first accessible GPU, or None if no GPU accessible.
    """
    for gpu_id in range(4):
        device_path = f"/dev/nvidia{gpu_id}"
        if os.path.exists(device_path):
            try:
                with open(device_path, 'rb') as f:
                    pass
                return gpu_id
            except (PermissionError, OSError):
                continue
    return None


def detect_all_accessible_gpus() -> List[int]:
    """
    Detect all accessible GPUs using device file approach.
    
    Returns
    -------
    List[int]
        List of accessible GPU IDs, empty if none accessible.
    """
    accessible_gpus = []
    for gpu_id in range(4):
        device_path = f"/dev/nvidia{gpu_id}"
        if os.path.exists(device_path):
            try:
                with open(device_path, 'rb') as f:
                    pass
                accessible_gpus.append(gpu_id)
            except (PermissionError, OSError):
                continue
    return accessible_gpus


def setup_cuda_environment(gpu_id: Optional[int] = None, verbose: bool = True) -> Optional[int]:
    """
    Setup CUDA environment variables for GPU usage.
    
    Parameters
    ----------
    gpu_id : int, optional
        Specific GPU ID to use. If None, auto-detect first accessible GPU.
    verbose : bool, optional
        Whether to print setup messages. Default True.
        
    Returns
    -------
    int or None
        GPU ID that was configured, or None if no GPU available.
    """
    # Auto-detect if not specified
    if gpu_id is None:
        gpu_id = detect_accessible_gpu()
    
    # Configure CUDA environment
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if verbose:
            print(f"✅ GPU {gpu_id} set for CUDA (CUDA_VISIBLE_DEVICES={gpu_id})")
        return gpu_id
    else:
        if verbose:
            print("⚠️  No GPU accessible - CUDA will use CPU fallback")
        return None


def test_gpu_computation(tf_module) -> tuple[bool, str]:
    """
    Test if GPU computation actually works with TensorFlow.
    
    Parameters
    ----------
    tf_module
        TensorFlow module (already imported).
        
    Returns
    -------
    tuple[bool, str]
        (success, device_name) where success indicates if GPU works,
        and device_name is '/GPU:0' or '/CPU:0'.
    """
    try:
        with tf_module.device('/GPU:0'):
            # Simple matrix multiplication test
            a = tf_module.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf_module.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf_module.matmul(a, b)
            result = c.numpy()
        return True, '/GPU:0'
    except Exception:
        return False, '/CPU:0'


def configure_gpu_memory_growth(tf_module, verbose: bool = True) -> bool:
    """
    Configure GPU memory growth for TensorFlow.
    
    Parameters
    ----------
    tf_module
        TensorFlow module (already imported).
    verbose : bool, optional
        Whether to print configuration messages. Default True.
        
    Returns
    -------
    bool
        True if memory growth was configured, False otherwise.
    """
    try:
        physical_gpus = tf_module.config.list_physical_devices('GPU')
        if physical_gpus:
            for gpu in physical_gpus:
                tf_module.config.experimental.set_memory_growth(gpu, True)
            if verbose:
                print(f"📋 GPU memory growth enabled for {len(physical_gpus)} GPU(s)")
            return True
        else:
            if verbose:
                print("📋 No physical GPUs detected - memory growth not configured")
            return False
    except Exception as e:
        if verbose:
            print(f"⚠️  Could not configure GPU memory growth: {e}")
        return False


def get_gpu_diagnostics(tf_module) -> Dict[str, Any]:
    """
    Get comprehensive GPU diagnostics information.
    
    Parameters
    ----------
    tf_module
        TensorFlow module (already imported).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing GPU diagnostic information.
    """
    diagnostics = {}
    
    # Environment variables
    diagnostics['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    # Device file accessibility
    diagnostics['accessible_gpus'] = detect_all_accessible_gpus()
    
    # TensorFlow GPU detection
    physical_gpus = tf_module.config.list_physical_devices('GPU')
    diagnostics['tf_physical_gpus'] = len(physical_gpus)
    diagnostics['tf_gpu_details'] = []
    
    for i, gpu in enumerate(physical_gpus):
        gpu_info = {'id': i, 'name': str(gpu)}
        try:
            details = tf_module.config.experimental.get_device_details(gpu)
            gpu_info['details'] = details
        except Exception as e:
            gpu_info['details'] = f"Not available: {e}"
        diagnostics['tf_gpu_details'].append(gpu_info)
    
    # Test GPU computation
    gpu_works, device = test_gpu_computation(tf_module)
    diagnostics['gpu_computation_works'] = gpu_works
    diagnostics['effective_device'] = device
    
    return diagnostics


def print_gpu_diagnostics(tf_module):
    """
    Print comprehensive GPU diagnostics to console.
    
    Parameters
    ----------
    tf_module
        TensorFlow module (already imported).
    """
    print("TensorFlow GPU Diagnostics")
    print("=" * 40)
    
    diagnostics = get_gpu_diagnostics(tf_module)
    
    # Environment
    print(f"CUDA_VISIBLE_DEVICES: {diagnostics['cuda_visible_devices']}")
    
    # Accessible GPUs (device files)
    accessible = diagnostics['accessible_gpus']
    if accessible:
        print(f"Accessible GPUs (device files): {accessible}")
    else:
        print("Accessible GPUs (device files): None")
    
    # TensorFlow GPU detection
    print(f"\nTensorFlow physical GPUs: {diagnostics['tf_physical_gpus']}")
    for gpu_info in diagnostics['tf_gpu_details']:
        print(f"  GPU {gpu_info['id']}: {gpu_info['name']}")
        print(f"    Details: {gpu_info['details']}")
    
    # GPU computation test
    print(f"\nGPU computation test:")
    if diagnostics['gpu_computation_works']:
        print(f"  ✅ GPU computation successful")
        print(f"  🎯 Effective device: {diagnostics['effective_device']}")
    else:
        print(f"  ❌ GPU computation failed")
        print(f"  🎯 Falling back to: {diagnostics['effective_device']}")
    
    print("=" * 40)


def setup_tensorflow_gpu(tf_module, gpu_id: Optional[int] = None, 
                        memory_growth: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    Complete TensorFlow GPU setup and configuration.
    
    Parameters
    ----------
    tf_module
        TensorFlow module (already imported).
    gpu_id : int, optional
        Specific GPU ID to use. If None, auto-detect.
    memory_growth : bool, optional
        Whether to enable GPU memory growth. Default True.
    verbose : bool, optional
        Whether to print setup messages. Default True.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'gpu_id': GPU ID used (or None)
        - 'device': Device string ('/GPU:0' or '/CPU:0')
        - 'gpu_available': bool indicating GPU availability
        - 'memory_growth_enabled': bool indicating if memory growth was set
    """
    # Setup CUDA environment
    configured_gpu_id = setup_cuda_environment(gpu_id, verbose)
    
    # Test GPU functionality
    gpu_works, device = test_gpu_computation(tf_module)
    
    # Configure memory growth if GPU is available
    memory_growth_enabled = False
    if gpu_works and memory_growth:
        memory_growth_enabled = configure_gpu_memory_growth(tf_module, verbose)
    
    # Print summary
    if verbose:
        print(f"🎯 Effective device: {device}")
        print(f"⚡ GPU acceleration: {'✅ Available' if gpu_works else '❌ CPU only'}")
    
    return {
        'gpu_id': configured_gpu_id,
        'device': device,
        'gpu_available': gpu_works,
        'memory_growth_enabled': memory_growth_enabled
    }


def check_pipeline_gpu_usage(pipeline_module) -> Dict[str, Any]:
    """
    Check if a pipeline module has GPU usage patterns.
    
    Parameters
    ----------
    pipeline_module
        Module to inspect for GPU usage.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with GPU usage analysis.
    """
    import inspect
    
    analysis = {
        'has_device_placement': False,
        'has_gpu_references': False,
        'functions_checked': []
    }
    
    try:
        # Get all functions in the module
        functions = [getattr(pipeline_module, name) for name in dir(pipeline_module) 
                    if callable(getattr(pipeline_module, name)) and not name.startswith('_')]
        
        for func in functions:
            func_name = getattr(func, '__name__', str(func))
            analysis['functions_checked'].append(func_name)
            
            try:
                source = inspect.getsource(func)
                
                # Check for device placement patterns
                device_patterns = ['tf.device', 'device=', '/GPU:', '/CPU:', 'with_device']
                if any(pattern in source for pattern in device_patterns):
                    analysis['has_device_placement'] = True
                
                # Check for GPU references
                gpu_patterns = ['GPU', 'gpu', 'cuda', 'CUDA']
                if any(pattern in source for pattern in gpu_patterns):
                    analysis['has_gpu_references'] = True
                    
            except (OSError, TypeError):
                # Can't get source (built-in function, etc.)
                continue
                
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis


def get_cluster_resource_summary() -> Dict[str, Any]:
    """
    Get comprehensive cluster resource allocation summary.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing cluster resource information.
    """
    summary = {
        'hostname': socket.gethostname(),
        'cpu_info': {},
        'memory_info': {},
        'gpu_info': {},
        'storage_quotas': []
    }
    
    # CPU Information
    try:
        import multiprocessing as mp
        logical_cores = mp.cpu_count()
        
        # Try to detect physical cores (more complex but accurate)
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Count unique physical IDs
            physical_cores = len(set([line.split(':')[1].strip() 
                                    for line in cpuinfo.split('\n') 
                                    if line.startswith('physical id')]))
            
            if physical_cores == 0:
                physical_cores = logical_cores
                
        except:
            physical_cores = logical_cores
        
        summary['cpu_info'] = {
            'logical_cores': logical_cores,
            'physical_cores': physical_cores,
            'has_hyperthreading': physical_cores != logical_cores
        }
        
        # Check for SLURM allocation
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_NTASKS')
        if slurm_cpus:
            summary['cpu_info']['allocated_cpus'] = int(slurm_cpus)
            summary['cpu_info']['scheduler'] = 'SLURM'
            
    except Exception as e:
        summary['cpu_info']['error'] = str(e)
    
    # Memory Information
    try:
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        # Check for SLURM memory limits
        slurm_memory = None
        slurm_mem_per_node = os.environ.get('SLURM_MEM_PER_NODE')
        if slurm_mem_per_node:
            slurm_memory = int(slurm_mem_per_node) / 1024  # MB to GB
        
        summary['memory_info'] = {
            'total_gb': total_memory_gb,
            'allocated_gb': slurm_memory if slurm_memory and slurm_memory < total_memory_gb else total_memory_gb
        }
        
    except Exception as e:
        summary['memory_info']['error'] = str(e)
    
    # GPU Information
    summary['gpu_info'] = {
        'accessible_gpus': detect_all_accessible_gpus(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    }
    
    # Storage Quotas (NYU Greene specific)
    try:
        summary['storage_quotas'] = get_nyu_storage_quotas()
    except Exception as e:
        summary['storage_quotas'] = {'error': str(e)}
    
    return summary


def get_nyu_storage_quotas() -> List[Dict[str, str]]:
    """
    Get NYU Greene storage quota information.
    
    Returns
    -------
    List[Dict[str, str]]
        List of storage quota information dictionaries.
    """
    import re
    
    try:
        cmd = ['ssh', '-o', 'ConnectTimeout=3', '-o', 'BatchMode=yes', 
               '-o', 'StrictHostKeyChecking=no', 'log-1.hpc.nyu.edu', 'myquota']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            
            # Clean ANSI color codes from output
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            
            quota_data = []
            for line in lines:
                line = ansi_escape.sub('', line).strip()
                if line.startswith('/'):
                    parts = line.split()
                    if len(parts) >= 5:
                        filesystem = parts[0]
                        allocation = parts[3]
                        usage_info = parts[4]
                        
                        # Extract usage and percentage
                        usage_parts = usage_info.split('/')
                        if usage_parts:
                            space_part = usage_parts[0]
                            # Extract usage and percentage using regex
                            match = re.search(r'([0-9.]+[KMGT]?B)\s*\(([0-9.]+)%\)', space_part)
                            if match:
                                used = match.group(1)
                                percent = f"{match.group(2)}%"
                                quota_data.append({
                                    'filesystem': filesystem,
                                    'allocation': allocation,
                                    'used': used,
                                    'percent': percent
                                })
            
            return quota_data
            
    except Exception:
        pass
    
    return []


def print_cluster_resource_summary():
    """
    Print a comprehensive cluster resource allocation summary.
    """
    summary = get_cluster_resource_summary()
    
    print("CLUSTER RESOURCE ALLOCATION SUMMARY")
    print("=" * 50)
    
    # Hostname
    print(f"Hostname: {summary['hostname']}")
    
    # CPU
    cpu_info = summary['cpu_info']
    if 'error' not in cpu_info:
        if cpu_info.get('has_hyperthreading', False):
            print(f"CPU cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical (hyperthreading)")
        else:
            print(f"CPU cores: {cpu_info['logical_cores']} logical")
        
        if 'allocated_cpus' in cpu_info:
            print(f"Job-allocated CPUs: {cpu_info['allocated_cpus']} ({cpu_info.get('scheduler', 'Unknown')})")
    
    # Memory
    memory_info = summary['memory_info']
    if 'error' not in memory_info:
        if memory_info['allocated_gb'] < memory_info['total_gb']:
            print(f"Job-allocated memory: {memory_info['allocated_gb']:.1f} GB")
        else:
            print(f"Memory: {memory_info['total_gb']:.1f} GB total")
    
    # GPU
    gpu_info = summary['gpu_info']
    accessible_gpus = gpu_info['accessible_gpus']
    if accessible_gpus:
        print(f"GPU: {len(accessible_gpus)} accessible")
        for gpu_id in accessible_gpus:
            print(f"  GPU {gpu_id}: /dev/nvidia{gpu_id}")
    else:
        print("GPU: Not accessible")
    
    # Storage quotas
    if summary['storage_quotas'] and 'error' not in summary['storage_quotas']:
        print("\nSTORAGE QUOTAS AND USAGE")
        print("=" * 50)
        quota_data = summary['storage_quotas']
        if quota_data:
            print(f"{'Filesystem':<12} {'Allocation':<15} {'Used':<12} {'Percent':<8}")
            print("-" * 50)
            for item in quota_data:
                print(f"{item['filesystem']:<12} {item['allocation']:<15} {item['used']:<12} {item['percent']:<8}")
    
    print("=" * 50)
    print("\nResource summary complete")


def get_gpu_monitoring_commands() -> Dict[str, str]:
    """
    Get GPU monitoring commands appropriate for the current environment.
    
    Returns
    -------
    Dict[str, str]
        Dictionary with monitoring command descriptions and commands.
    """
    commands = {}
    
    # Check for different GPU monitoring tools
    gpu_tools = [
        ('/usr/bin/nvidia-smi', 'nvidia-smi'),
        ('/usr/local/cuda/bin/nvidia-smi', 'nvidia-smi'),
        ('/opt/nvidia/bin/nvidia-smi', 'nvidia-smi'),
        ('nvidia-smi', 'nvidia-smi')  # In PATH
    ]
    
    nvidia_smi_available = False
    for tool_path, cmd in gpu_tools:
        try:
            result = subprocess.run([tool_path, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                nvidia_smi_available = True
                commands['nvidia_smi_basic'] = f"{tool_path}"
                commands['nvidia_smi_watch'] = f"watch -n 1 {tool_path}"
                commands['nvidia_smi_detailed'] = f"{tool_path} -q"
                break
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            continue
    
    if not nvidia_smi_available:
        commands['nvidia_smi_note'] = "nvidia-smi not available - check module loading: module avail cuda"
    
    # Alternative monitoring methods
    commands['gpu_device_check'] = "ls -la /dev/nvidia*"
    commands['cuda_env_check'] = "echo $CUDA_VISIBLE_DEVICES"
    commands['slurm_gpu_check'] = "echo $SLURM_GPUS_ON_NODE"
    
    return commands


def print_gpu_monitoring_guide():
    """
    Print a guide for GPU monitoring on the cluster.
    """
    print("GPU MONITORING GUIDE")
    print("=" * 40)
    
    commands = get_gpu_monitoring_commands()
    
    print("Available monitoring commands:")
    for desc, cmd in commands.items():
        if desc.endswith('_note'):
            print(f"⚠️  {cmd}")
        else:
            print(f"  {desc}: {cmd}")
    
    print("\nRecommended monitoring workflow:")
    print("1. Check GPU accessibility: ls -la /dev/nvidia*")
    print("2. Verify CUDA environment: echo $CUDA_VISIBLE_DEVICES")
    print("3. Monitor GPU usage during computation:")
    if 'nvidia_smi_watch' in commands:
        print(f"   {commands['nvidia_smi_watch']}")
    else:
        print("   Load CUDA module first: module load cuda")
        print("   Then run: watch -n 1 nvidia-smi")
    
    print("4. For detailed GPU info:")
    if 'nvidia_smi_detailed' in commands:
        print(f"   {commands['nvidia_smi_detailed']}")
    else:
        print("   nvidia-smi -q (after loading CUDA module)")
    
    print("=" * 40)
