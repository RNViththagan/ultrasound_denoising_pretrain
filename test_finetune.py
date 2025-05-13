import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim, seed_everything, print_gpu_info
from model import get_model
from datetime import datetime
import os
import argparse

def test_finetune(model, test_loader, config, num_samples, random_seed=None):
    """
    Test the fine-tuned Noisier2Noise (N2N) MedSeg U-Net model.

    Description:
    - Inputs: Pseudo-clean images (Y, original BUSI/HC18 images).
    - Outputs: Denoised images (Y_hat = f(Y)).
    - Objective: Evaluate denoising quality on test set using MSE loss, PSNR, and SSIM.
    - Outputs:
      - Console output: Average test loss, PSNR, SSIM.
      - Sample flow visualization for one test image (pseudo-clean, denoised).
      - Visualization saved in outs/<timestamp>/test_sample_flow_finetune_noise<noise_std>.png.
    - Notes:
      - Loads fine-tuned model from checkpoints/finetuned_unet_noise<noise_std>_final_<timestamp>.pth.
      - Tracks the same sample image as in training for consistency.
      - Supports multiple samples for visualization via num_samples.
    """
    if random_seed is not None:
        seed_everything(random_seed)

    model.eval()
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    sample_flow = None

    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing Finetuned")
        for _, input, img_path in loop:
            input = input.to(config.device)

            output = model(input)
            loss = loss_fn(output, input)
            test_loss += loss.item()
            test_psnr += calculate_psnr(loss).item()
            test_ssim += calculate_ssim(output, input).item()

            # Capture sample flow for the tracked image
            if sample_flow is None and config.data_dir in img_path:
                sample_flow = (output[0], input[0])

    avg_test_loss = test_loss / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)

    print(f"📊 Finetuned Test Results (Y_hat vs. Y, Noise Std={config.noise_std}):")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")

    # Visualize sample flow
    if sample_flow:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_flow[1].cpu().squeeze(), cmap='gray')
        plt.title("Pseudo-Clean")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(sample_flow[0].cpu().squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(config.output_dir, f"test_sample_flow_finetune_noise{config.noise_std}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📸 Saved test sample flow to {save_path}")

def main():
    from config import Config
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="Test fine-tuned model for ultrasound denoising")
    parser.add_argument('--random_seed', type=int, default=None,
                        help="Random seed for sample selection")
    parser.add_argument('--num_samples', type=int, default=None,
                        help="Number of sample images to visualize")
    args = parser.parse_args()

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()

    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    _, _, test_loader = get_dataloaders(config, mode='finetune')
    model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)

    checkpoint_path = os.path.join(config.checkpoint_dir, f"finetuned_unet_noise{config.noise_std}_final_{timestamp}.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"🧪 Testing on {len(test_loader.dataset)} test images")
    test_finetune(model, test_loader, config, args.num_samples or config.num_samples, args.random_seed)

if __name__ == "__main__":
    main()