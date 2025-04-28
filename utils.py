import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("❌ GPU not available. Using CPU.")

def calculate_psnr(mse_loss):
    if mse_loss == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse_loss)

def save_checkpoint(model, path, filename):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))