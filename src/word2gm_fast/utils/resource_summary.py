"""
System Resource Summary Utility

Provides comprehensive system resource information for debugging and monitoring
Word2GM training environmentdef print_resource_summary():
    """Print a comprehensive resource summary."""
    import time
    
    # Try using IPython display instead of print to avoid duplication
    try:
        from IPython.display import display
        use_display = True
    except ImportError:
        use_display = False
    
    def safe_print(text):
        if use_display:
            display(text)
        else:
            print(text)
    
    safe_print("SYSTEM RESOURCE SUMMARY")
    safe_print("=" * 50)ding SLURM job allocation details and GPU info.
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
    
    # Build the entire output as a single string to avoid multiple print calls
    output_lines = []
    output_lines.append("SYSTEM RESOURCE SUMMARY")
    output_lines.append("=" * 50)
    
    # Basic system info
    hostname = get_hostname()
    output_lines.append(f"Hostname: {hostname}")
    
    # SLURM job info
    slurm_info = get_slurm_info()
    output_lines.append("")
    output_lines.append("Job Allocation:")
    output_lines.append(f"   CPUs: {slurm_info['cpus']}")
    output_lines.append(f"   Memory: {slurm_info['memory_gb']:.1f} GB")
    output_lines.append(f"   Requested partitions: {slurm_info['requested_partitions']}")
    output_lines.append(f"   Actually running on: {slurm_info['actual_partition']}")
    output_lines.append(f"   Job ID: {slurm_info['job_id']}")
    output_lines.append(f"   Node list: {slurm_info['node_list']}")
    
    # GPU information
    gpu_info = get_gpu_info()
    output_lines.append("")
    output_lines.append("GPU Information:")
    
    if not gpu_info['available']:
        if 'error' in gpu_info:
            output_lines.append(f"   Error: {gpu_info['error']}")
        else:
            output_lines.append(f"   No CUDA GPUs detected")
    else:
        output_lines.append(f"   CUDA GPUs detected: {gpu_info['count']}")
        for device in gpu_info['devices']:
            output_lines.append(f"   GPU {device['index']}: {device['name']}")
            output_lines.append(f"      Memory: {device['memory_used_gb']:.1f}/"
                              f"{device['memory_total_gb']:.1f} GB "
                              f"({device['memory_free_gb']:.1f} GB free)")
            
            if device['temperature_c'] is not None:
                output_lines.append(f"      Temperature: {device['temperature_c']}°C")
            
            if device['gpu_utilization_percent'] is not None:
                output_lines.append(f"      Utilization: GPU {device['gpu_utilization_percent']}%, "
                                  f"Memory {device['memory_utilization_percent']}%")
    
    # TensorFlow GPU detection
    tf_info = get_tensorflow_info()
    output_lines.append("")
    output_lines.append("TensorFlow GPU Detection:")
    if 'error' in tf_info:
        output_lines.append(f"   Error: {tf_info['error']}")
    else:
        output_lines.append(f"   TensorFlow detects {tf_info['gpu_count']} GPU(s)")
        output_lines.append(f"   Built with CUDA: {tf_info['built_with_cuda']}")
        
        if tf_info['gpu_count'] > 0:
            for i, name in enumerate(tf_info['gpu_names']):
                growth = tf_info['memory_growth'][i] if i < len(tf_info['memory_growth']) else None
                growth_str = f", Memory growth: {growth}" if growth is not None else ""
                output_lines.append(f"      {name}{growth_str}")
    
    output_lines.append("=" * 50)
    
    # Single print statement to avoid duplication
    print("\n".join(output_lines))
    
    # Flush stdout to prevent output duplication
    import sys
    sys.stdout.flush()
    
    if not gpu_info['available']:
        if 'error' in gpu_info:
            time.sleep(0.50)
            print(f"   Error: {gpu_info['error']}")
        else:
            time.sleep(0.50)
            print(f"   No CUDA GPUs detected")
    else:
        time.sleep(0.50)
        print(f"   CUDA GPUs detected: {gpu_info['count']}")
        for device in gpu_info['devices']:
            time.sleep(0.50)
            print(f"   GPU {device['index']}: {device['name']}")
            time.sleep(0.50)
            print(f"      Memory: {device['memory_used_gb']:.1f}/"
                  f"{device['memory_total_gb']:.1f} GB "
                  f"({device['memory_free_gb']:.1f} GB free)")
            
            if device['temperature_c'] is not None:
                time.sleep(0.50)
                print(f"      Temperature: {device['temperature_c']}°C")
            
            if device['gpu_utilization_percent'] is not None:
                time.sleep(0.50)
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
