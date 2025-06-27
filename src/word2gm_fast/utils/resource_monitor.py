"""
Resource monitoring utilities for Word2GM training.

Provides a notebook-friendly ResourceMonitor class for periodic reporting of
GPU, CPU, and memory usage. Designed for integration with Jupyter/VS Code
notebooks and TensorBoard.
"""

import threading
import time
import os
import sys
import logging

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _has_nvml = True
except Exception:
    _has_nvml = False

import tensorflow as tf

class ResourceMonitor:
    """
    Periodically logs GPU, CPU, and memory usage to TensorBoard and/or stdout.
    """
    def __init__(self, log_dir=None, interval=10, summary_writer=None):
        self.interval = interval
        self.log_dir = log_dir
        self.summary_writer = summary_writer
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._step = 0
        self.log_fn = None  # Add log_fn attribute
        self.print_to_notebook = True  # Optionally control printing

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _run(self):
        while not self._stop_event.is_set():
            self.log_resource_usage()
            time.sleep(self.interval)
            self._step += 1

    def log_resource_usage(self):
        # CPU/memory
        cpu = psutil.cpu_percent() if psutil else None
        mem = psutil.virtual_memory().percent if psutil else None
        # GPU
        gpu_util, gpu_mem = None, None
        if _has_nvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = util.gpu
                gpu_mem = 100 * meminfo.used / meminfo.total
            except Exception:
                pass
        # Log to TensorBoard
        if self.summary_writer:
            with self.summary_writer.as_default():
                if cpu is not None:
                    tf.summary.scalar("resource/cpu_percent", cpu, step=self._step)
                if mem is not None:
                    tf.summary.scalar("resource/mem_percent", mem, step=self._step)
                if gpu_util is not None:
                    tf.summary.scalar("resource/gpu_util", gpu_util, step=self._step)
                if gpu_mem is not None:
                    tf.summary.scalar("resource/gpu_mem_percent", gpu_mem, step=self._step)
        # Print or log to notebook output
        msg = f"[Resource] Step {self._step}: "
        if cpu is not None:
            msg += f"CPU {cpu:.1f}% "
        if mem is not None:
            msg += f"Mem {mem:.1f}% "
        if gpu_util is not None:
            msg += f"GPU {gpu_util:.1f}% "
        if gpu_mem is not None:
            msg += f"GPU Mem {gpu_mem:.1f}% "
        if self.log_fn:
            self.log_fn(msg)
        elif getattr(self, 'print_to_notebook', True):
            print(msg)
