import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim, calculate_brisque, calculate_snr, calculate_cnr, calculate_gcnr, seed_everything, print_gpu_info
from model import get_model
from datetime import datetime
import os
import argparse
import glob

def test_finetune(model, test_loader, config, num_samples):
    """
    Test the fine-tuned Noisier2Noise (N2N) model with hybrid MSE-BRISQUE loss.

    Description:
    - Inputs: Original images (Y, from BUSI/HC18 datasets, assumed clean or minimally noisy).
    - Outputs: Denoised images (Y_hat = f(Y)).
    - Objective: Evaluate denoising quality on test set using total loss (MSE + weighted BRISQUE), MSE, PSNR, SSIM, BRISQUE, SNR, CNR, and gCNR.
    - Outputs:
      - Console output: Average metrics for Input (BRISQUE) and Denoised (Total Loss, MSE, PSNR, SSIM, BRISQUE, SNR, CNR, gCNR).
      - Sample visualization for `num_samples` test images (Input, Denoised).
      - Visualization saved in outs/<timestamp>/{model}/test_results_finetune_noise{noise_std}.png.
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
    test_snr = 0
    test_cnr = 0
    test_gcnr = 0
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
            test_snr += calculate_snr(output, input).item()
            test_cnr += calculate_cnr(output, input).item()
            test_gcnr += calculate_gcnr(output, input)

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse_loss = test_mse_loss / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)
    avg_brisque_input = test_brisque_input / len(test_loader)
    avg_brisque_denoised = test_brisque_denoised / len(test_loader)
    avg_test_snr = test_snr / len(test_loader)
    avg_test_cnr = test_cnr / len(test_loader)
    avg_test_gcnr = test_gcnr / len(test_loader)

    print(f"📊 Finetuned Test Results (Noise Std={config.noise_std}):")
    print("Input Metrics:")
    print(f"  Average BRISQUE: {avg_brisque_input:.2f}")
    print("Denoised Metrics:")
    print(f"  Average Total Loss (MSE + BRISQUE): {avg_test_loss:.4f}")
    print(f"  Average MSE Loss: {avg_test_mse_loss:.4f}")
    print(f"  Average PSNR: {avg_test_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_test_ssim:.4f}")
    print(f"  Average SNR: {avg_test_snr:.2f} dB")
    print(f"  Average CNR: {avg_test_cnr:.4f}")
    print(f"  Average gCNR: {avg_test_gcnr:.4f}")
    print(f"  Average BRISQUE: {avg_brisque_denoised:.2f}")

    # Select random samples for visualization
    dataset = test_loader.dataset
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    sample_images = []
    with torch.no_grad():
        for idx in sample_indices:
            _, input, _ = dataset[idx]  # Get original image (Y)
            input = input.unsqueeze(0).to(config.device)  # Add batch dimension
            output = model(input)  # Y_hat
            sample_images.append({
                'input': input[0].cpu().numpy().squeeze(),
                'denoised': output[0].cpu().numpy().squeeze()
            })

    # Visualize sample images
    if sample_images:
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
        if num_samples == 1:
            axes = [axes]  # Ensure axes is iterable for single sample
        for i, sample in enumerate(sample_images):
            # Input
            axes[i][0].imshow(sample['input'], cmap='gray')
            brisque_input = calculate_brisque(sample['input'])
            axes[i][0].set_title(f"Input\nBRISQUE: {brisque_input:.2f}", fontsize=10, pad=10)
            axes[i][0].axis('off')

            # Denoised output
            axes[i][1].imshow(sample['denoised'], cmap='gray')
            input_tensor = torch.tensor(sample['input']).unsqueeze(0).unsqueeze(0).detach()
            denoised_tensor = torch.tensor(sample['denoised']).unsqueeze(0).unsqueeze(0).detach()
            mse_loss = loss_fn(denoised_tensor, input_tensor)
            psnr = calculate_psnr(mse_loss).item()
            ssim = calculate_ssim(denoised_tensor, input_tensor).item()
            brisque_denoised = calculate_brisque(sample['denoised'])
            snr = calculate_snr(denoised_tensor, input_tensor).item()
            cnr = calculate_cnr(denoised_tensor, input_tensor).item()
            gcnr = calculate_gcnr(denoised_tensor, input_tensor)
            axes[i][1].set_title(
                f"Denoised\n"
                f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\nBRISQUE: {brisque_denoised:.2f}\n"
                f"SNR: {snr:.2f} dB\nCNR: {cnr:.4f}\ngCNR: {gcnr:.4f}",
                fontsize=10, pad=10
            )
            axes[i][1].axis('off')

        plt.tight_layout(pad=2.0)
        save_path = os.path.join(config.output_dir, f"test_results_finetune_noise{config.noise_std}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        print(f"📸 Saved sample visualizations to {save_path}")

def main():
    from config import Config
    from dataset import get_dataloaders

    parser = argparse.ArgumentParser(description="Test fine-tuned model for ultrasound denoising")
    parser.add_argument('--num_samples', type=int, default=4,
                        help="Number of sample images to visualize")
    parser.add_argument('--model', type=str, choices=['unet', 'resnet'], default='unet',
                        help="Model to test: 'unet' (MedSegUNet) or 'resnet' (ModifiedResNet)")
    args = parser.parse_args()

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()

    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp, args.model)
    config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    # Find the latest checkpoint
    checkpoint_pattern = os.path.join(config.checkpoint_dir, f"finetuned_{args.model}_noise*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No fine-tuned checkpoints found for {args.model} in {config.checkpoint_dir}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    print(f"✅ Using latest checkpoint: {checkpoint_path}")

    _, _, test_loader = get_dataloaders(config, mode='finetune')
    model = get_model(model_name=args.model, pretrained=(args.model == 'resnet'), pretrained_path=None).to(config.device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"🧪 Testing {args.model.upper()} on {len(test_loader.dataset)} test images")
    test_finetune(model, test_loader, config, args.num_samples)

if __name__ == "__main__":
    main()