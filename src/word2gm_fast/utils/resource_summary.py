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
from IPython.display import display, Markdown
import re


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
        slurm_info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
    
    # Partition information - get both requested and actual
    slurm_info['requested_partitions'] = os.environ.get('SLURM_JOB_PARTITION', 'unknown')
    slurm_info['job_id'] = os.environ.get('SLURM_JOB_ID', 'N/A')
    slurm_info['node_list'] = os.environ.get('SLURM_JOB_NODELIST', 'N/A')
    
    # Get actual partition by SSH'ing to login node
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        ssh_cmd = ['ssh', 'greene-login', 'squeue', '-j', job_id, '-h', '-o', '%P']
        for attempt in range(10):
            try:
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=20)
                
                if result.returncode == 0:
                    slurm_info['actual_partition'] = result.stdout.strip()
                    break
                
                # Check for host key warning
                if "REMOTE HOST IDENTIFICATION HAS CHANGED" in result.stderr:
                    match = re.search(r"Offending (?:ECDSA|RSA|ED25519) key in (.*):(\d+)", result.stderr)
                    if match:
                        known_hosts_path = match.group(1)
                        line_number = int(match.group(2))
                        
                        # Remove the offending line
                        with open(known_hosts_path, "r") as f:
                            lines = f.readlines()
                        with open(known_hosts_path, "w") as f:
                            for i, line in enumerate(lines, 1):
                                if i != line_number:
                                    f.write(line)
                        continue  # Retry SSH
                    
                    else:
                        slurm_info['actual_partition'] = "SSH host key error: could not parse offending line"
                        break
                
                else:
                    slurm_info['actual_partition'] = f"SSH failed: {result.stderr.strip()}"
                    break
            
            except subprocess.TimeoutExpired:
                slurm_info['actual_partition'] = "SSH timeout"
                break
            except Exception as e:
                slurm_info['actual_partition'] = f"SSH error: {e}"
                break
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
    """Print a comprehensive resource summary using monospace font in notebook."""
    output_lines = []
    output_lines.append("SYSTEM RESOURCE SUMMARY")
    output_lines.append("=" * 60)
    
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
    output_lines.append(f"   Running on: {slurm_info['actual_partition']}")
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
        output_lines.append(f"   TensorFlow detects {tf_info.get('gpu_devices', 'N/A')} GPU(s)")
        if tf_info.get('gpu_devices', 0) > 0:
            for i, name in enumerate(tf_info.get('gpu_names', [])):
                growth = tf_info.get('memory_growth', [None]*tf_info.get('gpu_devices', 0))[i] if 'memory_growth' in tf_info else None
                growth_str = f", Memory growth: {growth}" if growth is not None else ""
                output_lines.append(f"      {name}{growth_str}")
        output_lines.append(f"   Built with CUDA: {tf_info.get('built_with_cuda', 'N/A')}")
    
    output_lines.append("=" * 60)
    
    # Display as monospace, no background
    display(Markdown(f"<pre>{'\n'.join(output_lines)}</pre>"))
    
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
