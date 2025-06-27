import pytest
import time
import sys

# Import the ResourceMonitor class
from src.word2gm_fast.utils.resource_monitor import ResourceMonitor

@pytest.mark.timeout(30)
def test_resource_monitor_basic():
    """
    Test that ResourceMonitor starts, logs at least once, and stops cleanly.
    """
    logs = []
    def log_fn(msg):
        logs.append(msg)
    
    monitor = ResourceMonitor(interval=2, summary_writer=None)
    monitor.print_to_notebook = False  # Silence direct print
    monitor.log_fn = log_fn  # Custom log function
    monitor.start()
    time.sleep(5)  # Let it log at least twice
    monitor.stop()
    
    # Check that logs were collected
    assert len(logs) >= 2, f"Expected at least 2 logs, got {len(logs)}"
    # Check that log messages contain expected resource info
    assert any("CPU" in msg or "GPU" in msg for msg in logs), "No resource info in logs"

@pytest.mark.timeout(30)
def test_resource_monitor_tensorboard(tmp_path):
    """
    Test that ResourceMonitor can write to a TensorBoard log directory.
    """
    from tensorflow.summary import create_file_writer
    log_dir = tmp_path / "tb_logs"
    writer = create_file_writer(str(log_dir))
    monitor = ResourceMonitor(interval=2, summary_writer=writer)
    monitor.print_to_notebook = False
    monitor.start()
    time.sleep(5)
    monitor.stop()
    # Check that TensorBoard log files were created
    assert any(log_dir.iterdir()), "No TensorBoard logs created"
