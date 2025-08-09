import torch
import random
import numpy as np
import os
from pytorch_ssim import SSIM
import math
from scipy import ndimage
from scipy import special
import warnings

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

def calculate_brisque(image):
    """Calculate BRISQUE score for a single grayscale image (numpy array in [0, 1])."""
    # Ensure image is in [0, 255] and uint8
    image = (image * 255).clip(0, 255).astype(np.uint8)
    # Ensure 2D grayscale image
    if len(image.shape) == 3:
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
        else:
            raise ValueError(f"Expected grayscale or single-channel image, got shape {image.shape}")
    elif len(image.shape) != 2:
        raise ValueError(f"Expected grayscale image, got shape {image.shape}")

    def generalized_gaussian_dist_fit(data):
        """Fit GGD to data and return shape parameter."""
        data = data.ravel()
        mean = np.mean(data)
        variance = np.var(data)
        if variance == 0:
            return 0.5  # Default shape to avoid division by zero
        gamma = np.log(2) / np.log(np.mean(np.abs(data - mean)**2) / variance)
        return max(0.2, min(gamma, 10.0))  # Constrain shape for stability

    def asymmetric_gaussian_dist_fit(data):
        """Fit AGGD to data and return parameters (alpha, beta_l, beta_r)."""
        data = data.ravel()
        mean = np.mean(data)
        left_data = data[data < mean]
        right_data = data[data >= mean]
        variance_left = np.var(left_data) if len(left_data) > 0 else 1.0
        variance_right = np.var(right_data) if len(right_data) > 0 else 1.0
        alpha = generalized_gaussian_dist_fit(data)
        beta_l = np.sqrt(variance_left) if variance_left > 0 else 1.0
        beta_r = np.sqrt(variance_right) if variance_right > 0 else 1.0
        return alpha, beta_l, beta_r

    def compute_mscn(image, kernel_size=7, sigma=7/6):
        """Compute MSCN coefficients."""
        img = image.astype(np.float32)
        mu = ndimage.gaussian_filter(img, sigma=sigma, mode='reflect')
        mu_sq = mu * mu
        sigma = np.sqrt(np.abs(ndimage.gaussian_filter(img * img, sigma=sigma, mode='reflect') - mu_sq))
        sigma = np.clip(sigma, 1e-10, None)  # Avoid division by zero
        mscn = (img - mu) / sigma
        return mscn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mscn = compute_mscn(image)
        features = []
        shape = generalized_gaussian_dist_fit(mscn)
        features.append(shape)
        features.append(np.var(mscn))
        pairwise = [
            mscn[:-1, :-1] * mscn[:-1, 1:],
            mscn[:-1, :-1] * mscn[1:, :-1],
            mscn[:-1, :-1] * mscn[1:, 1:],
            mscn[:-1, 1:] * mscn[1:, :-1]
        ]
        for pair in pairwise:
            alpha, beta_l, beta_r = asymmetric_gaussian_dist_fit(pair)
            features.extend([alpha, beta_l, beta_r, (beta_l + beta_r) / 2])
        downscaled = image[::2, ::2]
        mscn_down = compute_mscn(downscaled)
        shape_down = generalized_gaussian_dist_fit(mscn_down)
        features.append(shape_down)
        features.append(np.var(mscn_down))
        pairwise_down = [
            mscn_down[:-1, :-1] * mscn_down[:-1, 1:],
            mscn_down[:-1, :-1] * mscn_down[1:, :-1],
            mscn_down[:-1, :-1] * mscn_down[1:, 1:],
            mscn_down[:-1, 1:] * mscn_down[1:, :-1]
        ]
        for pair in pairwise_down:
            alpha, beta_l, beta_r = asymmetric_gaussian_dist_fit(pair)
            features.extend([alpha, beta_l, beta_r, (beta_l + beta_r) / 2])
        score = np.mean(features[1::4]) * 100
        return np.clip(score, 0, 100)

def calculate_mscn_variance(images, kernel_size=7, sigma=7/6):
    """Compute differentiable MSCN variance for a batch of images."""
    images = images.float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=images.device) / (kernel_size**2)
    mu = torch.nn.functional.conv2d(images, kernel, padding=kernel_size//2)
    mu_sq = mu * mu
    sigma = torch.sqrt(torch.abs(torch.nn.functional.conv2d(images * images, kernel, padding=kernel_size//2) - mu_sq))
    sigma = torch.clamp(sigma, min=1e-10)
    mscn = (images - mu) / sigma
    mscn_variance = torch.var(mscn, dim=(2, 3), keepdim=False).mean(dim=1)
    return mscn_variance.mean()

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