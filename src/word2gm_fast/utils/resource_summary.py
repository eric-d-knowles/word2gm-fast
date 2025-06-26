"""
System Resource Summary Utility

Provides comprehensive system resource information for debugging and monitoring
Word2GM training environments, including SLURM job allocation details and GPU info.
"""

import os
import socket
import subprocess
import psutil
import pynvml
import tensorflow as tf


def get_hostname():
    """Get system hostname."""
    return socket.gethostname()


def get_slurm_info():
    """Get SLURM job allocation information."""
    slurm_info = {}
    
    # CPU allocation
    slurm_info['cpus'] = int(os.environ.get(
        'SLURM_CPUS_PER_TASK', 
        psutil.cpu_count()
    ))
    
    # Memory allocation
    mem_per_node_mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mem_per_node_mb:
        slurm_info['memory_gb'] = int(mem_per_node_mb) / 1024
    else:
        # Fallback to system memory
        slurm_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    
    # Partition information - get both requested and actual
    slurm_info['requested_partitions'] = os.environ.get('SLURM_JOB_PARTITION', 'unknown')
    slurm_info['job_id'] = os.environ.get('SLURM_JOB_ID', 'N/A')
    slurm_info['node_list'] = os.environ.get('SLURM_JOB_NODELIST', 'N/A')
    
    # Get actual partition by SSH'ing to login node
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        try:
            # SSH to greene-login and run squeue to get actual partition
            ssh_cmd = ['ssh', 'greene-login', 'squeue', '-j', job_id, '-h', '-o', '%P']
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                slurm_info['actual_partition'] = result.stdout.strip()
            else:
                slurm_info['actual_partition'] = f"SSH failed: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            slurm_info['actual_partition'] = "SSH timeout"
        except Exception as e:
            slurm_info['actual_partition'] = f"SSH error: {e}"
    else:
        slurm_info['actual_partition'] = 'N/A (not in SLURM job)'
    
    return slurm_info


def get_gpu_info():
    """Get GPU information using pynvml."""
    gpu_info = {'available': False, 'devices': []}
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info['available'] = device_count > 0
        gpu_info['count'] = device_count
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature if available
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp = None
            
            # Get utilization if available
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
            except:
                gpu_util = None
                memory_util = None
            
            device_info = {
                'index': i,
                'name': name,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_free_gb': memory_info.free / (1024**3),
                'temperature_c': temp,
                'gpu_utilization_percent': gpu_util,
                'memory_utilization_percent': memory_util
            }
            gpu_info['devices'].append(device_info)
            
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info


def get_tensorflow_info():
    """Get TensorFlow GPU detection information."""
    tf_info = {}
    
    try:
        # GPU devices detected by TensorFlow
        physical_devices = tf.config.list_physical_devices('GPU')
        tf_info['gpu_devices'] = len(physical_devices)
        tf_info['gpu_names'] = [device.name for device in physical_devices]
        
        # Memory growth setting
        if physical_devices:
            tf_info['memory_growth'] = []
            for device in physical_devices:
                try:
                    growth = tf.config.experimental.get_memory_growth(device)
                    tf_info['memory_growth'].append(growth)
                except:
                    tf_info['memory_growth'].append(None)
        
        # CUDA version info
        tf_info['built_with_cuda'] = tf.test.is_built_with_cuda()
        
    except Exception as e:
        tf_info['error'] = str(e)
    
    return tf_info


def print_resource_summary():
    """Print a comprehensive resource summary."""
    import time
    
    time.sleep(0.01)
    print("SYSTEM RESOURCE SUMMARY")
    time.sleep(0.01)
    print("=" * 50)
    
    # Basic system info
    hostname = get_hostname()
    print(f"Hostname: {hostname}")
    
    # SLURM job info
    slurm_info = get_slurm_info()
    print(f"\nJob Allocation:")
    print(f"   CPUs: {slurm_info['cpus']}")
    print(f"   Memory: {slurm_info['memory_gb']:.1f} GB")
    print(f"   Partition: {slurm_info['actual_partition']}")
    print(f"   Job ID: {slurm_info['job_id']}")
    print(f"   Node list: {slurm_info['node_list']}")
    
    # GPU information
    gpu_info = get_gpu_info()
    print(f"\nGPU Information:")
    
    if not gpu_info['available']:
        if 'error' in gpu_info:
            print(f"   Error: {gpu_info['error']}")
        else:
            print(f"   No CUDA GPUs detected")
    else:
        print(f"   CUDA GPUs detected: {gpu_info['count']}")
        for device in gpu_info['devices']:
            print(f"   GPU {device['index']}: {device['name']}")
            print(f"      Memory: {device['memory_used_gb']:.1f}/"
                  f"{device['memory_total_gb']:.1f} GB "
                  f"({device['memory_free_gb']:.1f} GB free)")
            
            if device['temperature_c'] is not None:
                print(f"      Temperature: {device['temperature_c']}°C")
            
            if device['gpu_utilization_percent'] is not None:
                print(f"      Utilization: GPU {device['gpu_utilization_percent']}%, "
                      f"Memory {device['memory_utilization_percent']}%")
    
    # TensorFlow GPU detection
    tf_info = get_tensorflow_info()
    print(f"\nTensorFlow GPU Detection:")
    
    if 'error' in tf_info:
        print(f"   Error: {tf_info['error']}")
    else:
        print(f"   TensorFlow detects {tf_info['gpu_devices']} GPU(s)")
        print(f"   Built with CUDA: {tf_info['built_with_cuda']}")
        
        if tf_info['gpu_devices'] > 0:
            for i, name in enumerate(tf_info['gpu_names']):
                growth = tf_info['memory_growth'][i] if i < len(tf_info['memory_growth']) else None
                growth_str = f", Memory growth: {growth}" if growth is not None else ""
                print(f"      {name}{growth_str}")
    
    print("=" * 50)
    
    # Flush stdout to prevent output duplication
    import sys
    sys.stdout.flush()


def get_resource_dict():
    """Return resource information as a dictionary for programmatic use."""
    return {
        'hostname': get_hostname(),
        'slurm': get_slurm_info(),
        'gpu': get_gpu_info(),
        'tensorflow': get_tensorflow_info()
    }
