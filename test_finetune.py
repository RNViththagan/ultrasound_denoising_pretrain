import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim, calculate_brisque, seed_everything, print_gpu_info
from model import get_model
from datetime import datetime
import os
import argparse
import glob

def test_finetune(model, test_loader, config, num_samples):
    """
    Test the fine-tuned Noisier2Noise (N2N) MedSeg U-Net model with hybrid MSE-BRISQUE loss.

    Description:
    - Inputs: Pseudo-clean images (Y, original BUSI/HC18 images).
    - Outputs: Denoised images (Y_hat = f(Y)).
    - Objective: Evaluate denoising quality on test set using total loss (MSE + weighted BRISQUE), MSE, PSNR, SSIM, and BRISQUE.
    - Outputs:
      - Console output: Average test total loss, MSE loss, PSNR, SSIM, BRISQUE (input and denoised).
      - Sample visualization for `num_samples` test images (input, denoised).
      - Visualization saved in outs/<timestamp>/test_results_finetune_noise{noise_std}.png.
    - Notes:
      - Loads fine-tuned model from the latest checkpoint in checkpoints/.
      - Uses random sampling for visualization.
      - BRISQUE is used as a non-differentiable regularizer in the total loss, consistent with training.
    """
    seed_everything(config.random_seed)

    model.eval()
    test_loss = 0
    test_mse_loss = 0
    test_psnr = 0
    test_ssim = 0
    test_brisque_input = 0
    test_brisque_denoised = 0
    brisque_weight = 0.01  # Same as in finetune.py

    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing Finetuned")
        for _, input, _ in loop:  # Ignore doubly_noisy, use input (Y)
            input = input.to(config.device)

            output = model(input)  # Y_hat = f(Y)
            mse_loss = loss_fn(output, input)
            brisque_score = calculate_brisque(output[0].detach().cpu().numpy().squeeze())
            brisque_loss = brisque_score * brisque_weight
            total_loss = mse_loss + brisque_loss

            test_loss += total_loss
            test_mse_loss += mse_loss.item()
            test_psnr += calculate_psnr(mse_loss).item()
            test_ssim += calculate_ssim(output, input).item()
            test_brisque_input += calculate_brisque(input[0].detach().cpu().numpy().squeeze())
            test_brisque_denoised += brisque_score

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse_loss = test_mse_loss / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)
    avg_brisque_input = test_brisque_input / len(test_loader)
    avg_brisque_denoised = test_brisque_denoised / len(test_loader)

    print(f"📊 Finetuned Test Results (Y_hat vs. Y, Noise Std={config.noise_std}):")
    print(f"Average Test Total Loss (MSE + BRISQUE): {avg_test_loss:.4f}")
    print(f"Average Test MSE Loss: {avg_test_mse_loss:.4f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")
    print(f"Average BRISQUE (Input): {avg_brisque_input:.2f}")
    print(f"Average BRISQUE (Denoised): {avg_brisque_denoised:.2f}")

    # Select random samples for visualization
    dataset = test_loader.dataset
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sample_images = []
    with torch.no_grad():
        for idx in sample_indices:
            _, input, _ = dataset[idx]  # Get pseudo-clean image (Y)
            input = input.unsqueeze(0).to(config.device)  # Add batch dimension
            output = model(input)  # Y_hat
            sample_images.append({
                'input': input[0].cpu().numpy().squeeze(),
                'denoised': output[0].cpu().numpy().squeeze()
            })

    # Visualize sample images
    if sample_images:
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]  # Ensure axes is iterable for single sample
        for i, sample in enumerate(sample_images):
            # Input (pseudo-clean)
            axes[i][0].imshow(sample['input'], cmap='gray')
            brisque_input = calculate_brisque(sample['input'])
            axes[i][0].set_title(f"Input (Pseudo-Clean, BRISQUE: {brisque_input:.2f})")
            axes[i][0].axis('off')

            # Denoised output
            axes[i][1].imshow(sample['denoised'], cmap='gray')
            # Calculate per-sample PSNR, SSIM, BRISQUE
            input_tensor = torch.tensor(sample['input']).unsqueeze(0).unsqueeze(0).detach()
            denoised_tensor = torch.tensor(sample['denoised']).unsqueeze(0).unsqueeze(0).detach()
            mse_loss = loss_fn(denoised_tensor, input_tensor)
            psnr = calculate_psnr(mse_loss).item()
            ssim = calculate_ssim(denoised_tensor, input_tensor).item()
            brisque_denoised = calculate_brisque(sample['denoised'])
            axes[i][1].set_title(f"Denoised (PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, BRISQUE: {brisque_denoised:.2f})")
            axes[i][1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(config.output_dir, f"test_results_finetune_noise{config.noise_std}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📸 Saved sample visualizations to {save_path}")

def main():
    from config import Config
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="Test fine-tuned model for ultrasound denoising")
    parser.add_argument('--num_samples', type=int, default=4,
                        help="Number of sample images to visualize")
    args = parser.parse_args()

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()

    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    # Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, "finetuned_unet_noise*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No fine-tuned checkpoints found in {config.checkpoint_dir}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"✅ Using latest checkpoint: {checkpoint_path}")

    _, _, test_loader = get_dataloaders(config, mode='finetune')
    model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"🧪 Testing on {len(test_loader.dataset)} test images")
    test_finetune(model, test_loader, config, args.num_samples)

if __name__ == "__main__":
    main()