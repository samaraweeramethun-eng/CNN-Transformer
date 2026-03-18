import torch


def setup_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for idx in range(device_count):
            gpu_name = torch.cuda.get_device_name(idx)
            memory_gb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
            print(f"GPU {idx}: {gpu_name} ({memory_gb:.1f} GB)")
        device = torch.device("cuda:0")
        return device, device_count > 1
    print("CUDA not available, using CPU")
    return torch.device("cpu"), False
