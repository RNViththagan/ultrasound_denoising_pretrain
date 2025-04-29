import torch
import math
import random
import numpy as np
import pytorch_ssim  # For SSIM computation

def print_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name}")
    else:
        print("⚠️ No GPU available. Using CPU.")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_psnr(mse):
    """
    Calculate PSNR given MSE.
    PSNR = 10 * log10(1 / MSE)
    """
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(img1, img2, mask=None):
    """
    Calculate SSIM between img1 and img2. If mask is provided, compute SSIM only on unmasked pixels.
    img1, img2: [B, C, H, W], values in [0, 1]
    mask: [B, C, H, W], 1 for unmasked, 0 for masked (optional)
    Returns: Average SSIM score across batch
    """
    if mask is not None:
        # Apply mask to images
        img1 = img1 * mask
        img2 = img2 * mask
    ssim_value = pytorch_ssim.ssim(img1, img2, window_size=11, size_average=True)
    return ssim_value

def save_checkpoint(model, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved checkpoint to {save_path}")