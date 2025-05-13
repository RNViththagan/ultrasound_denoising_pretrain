import torch
import random
import numpy as np
import os
from pytorch_ssim import SSIM
import math

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_psnr(loss):
    """Calculate PSNR from MSE loss."""
    mse = loss
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(pred, target):
    """Calculate SSIM between predicted and target images."""
    ssim = SSIM(window_size=11, size_average=True)
    return ssim(pred, target)

def save_checkpoint(model, checkpoint_dir, filename):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")

def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
    else:
        print("🖥️ Using CPU")