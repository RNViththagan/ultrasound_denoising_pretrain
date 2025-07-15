import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim, calculate_brisque, save_checkpoint
from model import get_model
from datetime import datetime
import os

def finetune(model, train_loader, val_loader, config, skip_pretrain=False):
    """
    Fine-tune the MedSeg U-Net using Noisier2Noise (N2N) for ultrasound denoising.

    Description:
    - Inputs: Doubly-noisy images (Z = Y + Y*M, where Y is pseudo-clean, M is noise with std=0.1).
    - Targets: Pseudo-clean images (Y, original BUSI/HC18 images).
    - Objective: Minimize MSE loss to denoise images.
    - Modes:
      - With pretrained: Load N2V-pretrained weights (from train.py).
      - From scratch: Use random initial weights.
    - Outputs:
      - Fine-tuned model checkpoints saved in checkpoints/ (every 10 epochs and final).
      - Sample flow visualization for one test image (doubly-noisy, denoised, pseudo-clean).
      - Metrics plots (loss, PSNR, SSIM vs. epochs) saved in outs/<timestamp>/.
      - Console output: Per-epoch training metrics (loss, PSNR, SSIM, BRISQUE).
    - Notes:
      - No validation (70/30 train/test split).
      - Sample image is tracked to show denoising quality.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.finetune_lr)

    train_losses, train_psnrs, train_ssims, train_brisques = [], [], [], []
    sample_flow = None

    init_type = "Random weights" if skip_pretrain else "Noise2Void pretrained"
    print(f"🟢 Starting fine-tuning with {init_type} weights (Noisier2Noise).")

    for epoch in range(config.finetune_epochs):
        model.train()
        running_loss = 0
        running_psnr = 0
        running_ssim = 0
        running_brisque = 0
        n_brisque_samples = 0

        loop = tqdm(train_loader, desc=f"[Finetune Epoch {epoch+1}/{config.finetune_epochs}]")
        for doubly_noisy, input, img_path in loop:
            doubly_noisy, input = doubly_noisy.to(config.device), input.to(config.device)

            output = model(doubly_noisy)
            loss = loss_fn(output, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(loss).item()
            ssim = calculate_ssim(output, input).item()
            running_loss += loss.item()
            running_psnr += psnr
            running_ssim += ssim

            # Compute BRISQUE every 10 epochs for first image in batch
            if (epoch + 1) % 10 == 0:
                brisque = calculate_brisque(output[0].detach().cpu().numpy().squeeze())
                running_brisque += brisque
                n_brisque_samples += 1

            loop.set_postfix(loss=loss.item(), psnr=psnr, ssim=ssim)

            # Capture sample flow for the tracked image
            if sample_flow is None and config.data_dir in img_path[0]:
                sample_flow = (doubly_noisy[0], output[0], input[0])

        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_ssim = running_ssim / len(train_loader)
        avg_train_brisque = running_brisque / n_brisque_samples if n_brisque_samples > 0 else float('inf')

        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        train_ssims.append(avg_train_ssim)
        train_brisques.append(avg_train_brisque if n_brisque_samples > 0 else None)

        log_msg = (f"📊 Epoch {epoch+1}/{config.finetune_epochs} | "
                   f"Train Loss: {avg_train_loss:.4f} | PSNR: {avg_train_psnr:.2f} | SSIM: {avg_train_ssim:.4f}")
        if n_brisque_samples > 0:
            log_msg += f" | BRISQUE: {avg_train_brisque:.2f}"
        print(log_msg)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = config._timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"finetuned_unet_noise{config.noise_std}_epoch{epoch+1}_{timestamp}.pth"
            save_checkpoint(model, config.checkpoint_dir, filename)

    # Save final model
    timestamp = config._timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_filename = f"finetuned_unet_noise{config.noise_std}_final_{timestamp}.pth"
    save_checkpoint(model, config.checkpoint_dir, final_filename)
    print(f"✅ Saved final fine-tuned model to {final_filename}")

    # Visualize sample flow
    if sample_flow:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_flow[0].detach().cpu().squeeze(), cmap='gray')
        plt.title("Doubly-Noisy Input")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sample_flow[1].detach().cpu().squeeze(), cmap='gray')
        brisque_denoised = calculate_brisque(sample_flow[1].detach().cpu().numpy().squeeze())
        plt.title(f"Denoised Output (BRISQUE: {brisque_denoised:.2f})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(sample_flow[2].cpu().squeeze(), cmap='gray')
        brisque_input = calculate_brisque(sample_flow[2].detach().cpu().numpy().squeeze())
        plt.title(f"Pseudo-Clean (BRISQUE: {brisque_input:.2f})")
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(config.output_dir, f"sample_flow_finetune_noise{config.noise_std}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📸 Saved fine-tuning sample flow to {save_path}")

    # Plot metrics
    from train import plot_metrics
    plot_metrics(train_losses, train_psnrs, train_ssims, config.output_dir, f"finetune_noise{config.noise_std}")

def main():
    from config import Config
    from dataset import get_dataloaders
    from utils import seed_everything, print_gpu_info

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    seed_everything(42)

    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, _, test_loader = get_dataloaders(config, mode='finetune')
    model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)
    finetune(model, train_loader, None, config, skip_pretrain=True)

if __name__ == "__main__":
    main()