# utils/helpers.py

import yaml
import os
import json
import kagglehub
import torch
import pynvml

def load_config(config_path):
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None

def get_model_from_Kaggle(kaggle_handle):
    """
    Downloads a model from Kaggle Hub using the provided handle.
    Requires Kaggle API credentials to be set up (~/.kaggle/kaggle.json).

    Args:
        kaggle_handle (str): The Kaggle Hub model handle (e.g., 'google/yolov10/pyTorch/yolov10s').

    Returns:
        str: The local path where the model was downloaded, or None if download failed.
    """
    try:
        # Ensure Kaggle API credentials are set from ~/.kaggle/kaggle.json
        # The kagglehub library usually handles this automatically if the file exists.
        print(f"Attempting to download model from Kaggle Hub: {kaggle_handle}")
        model_path = kagglehub.model_download(handle=kaggle_handle)
        print(f"Model successfully downloaded to: {model_path}")
        return model_path
    except FileNotFoundError:
        print("Error: Kaggle API credentials (~/.kaggle/kaggle.json) not found.")
        print("Please ensure your Kaggle API token is correctly set up.")
        return None
    except Exception as e:
        print(f"An error occurred while downloading the model from Kaggle Hub: {e}")
        return None

class MemoryTracker:
    """
    A simple utility class to track GPU memory usage using pynvml.

    Attributes:
        device (torch.device): The GPU device to monitor.
        handle: Handle to the GPU obtained from pynvml.
        initial_used_mem (int): GPU memory used (in bytes) when the tracker was initialized.
    """
    def __init__(self, device):
        """
        Initializes the MemoryTracker for a specific GPU device.

        Args:
            device (torch.device or int): The GPU device (e.g., torch.device('cuda:0') or 0).
        """
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. MemoryTracker will not function.")
            self.handle = None
            self.initial_used_mem = 0
            return

        try:
            pynvml.nvmlInit()
            device_index = device.index if isinstance(device, torch.device) else device
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.initial_used_mem = info.used
            print(f"MemoryTracker initialized for device {device_index}. Initial used memory: {self.initial_used_mem / (1024**2):.2f} MB")
        except pynvml.NVMLError as error:
            print(f"Failed to initialize NVML: {error}. Memory tracking disabled.")
            self.handle = None
            self.initial_used_mem = 0
        except Exception as e:
            print(f"An unexpected error occurred during MemoryTracker initialization: {e}. Memory tracking disabled.")
            self.handle = None
            self.initial_used_mem = 0

    def get_used_memory_bytes(self):
        """Returns the currently used GPU memory in bytes."""
        if self.handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return info.used
            except pynvml.NVMLError as error:
                print(f"Failed to get memory info: {error}")
                return 0
        return 0

    def get_used_memory_mb(self):
        """Returns the currently used GPU memory in megabytes."""
        return self.get_used_memory_bytes() / (1024**2)

    def get_memory_usage_since_init_mb(self):
        """Returns the GPU memory usage increase (in MB) since initialization."""
        current_used = self.get_used_memory_bytes()
        return (current_used - self.initial_used_mem) / (1024**2)

    def __del__(self):
        """Ensures NVML is shut down when the object is destroyed."""
        if self.handle:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as error:
                # Can sometimes happen if shutdown multiple times, usually safe to ignore
                # print(f"NVML Shutdown error: {error}")
                pass