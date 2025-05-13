import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim, save_checkpoint
from model import get_model
from datetime import datetime
import os

def pretrain(model, train_loader, val_loader, config):
    """
    Pretrain the MedSeg U-Net using Noise2Void (N2V) for self-supervised masked image reconstruction.

    Description:
    - Inputs: Masked images (X * M, where M is a binary mask with 10% pixels set to 0).
    - Targets: Original images (X), with loss computed only on unmasked pixels.
    - Objective: Minimize MSE loss on unmasked pixels to reconstruct the original image.
    - Outputs:
      - Trained model checkpoints saved in checkpoints/ (every 10 epochs and final).
      - Sample flow visualization for one test image (masked, reconstructed, original).
      - Metrics plots (loss, PSNR, SSIM vs. epochs) saved in outs/<timestamp>/.
      - Console output: Per-epoch training metrics (loss, PSNR, SSIM).
    - Notes:
      - Uses random initial weights (no pretrained weights).
      - No validation (70/30 train/test split).
      - Sample image is tracked to show reconstruction quality.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)

    train_losses, train_psnrs, train_ssims = [], [], []
    sample_flow = None

    print("🟢 Starting pretraining with random initial weights (Noise2Void).")

    for epoch in range(config.pretrain_epochs):
        model.train()
        running_loss = 0
        running_psnr = 0
        running_ssim = 0

        loop = tqdm(train_loader, desc=f"[Pretrain Epoch {epoch+1}/{config.pretrain_epochs}]")
        for masked_image, original_image, mask, img_path in loop:
            masked_image = masked_image.to(config.device)
            original_image = original_image.to(config.device)
            mask = mask.to(config.device)

            output = model(masked_image)
            loss = loss_fn(output * mask, original_image * mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(loss).item()
            ssim = calculate_ssim(output * mask, original_image * mask).item()
            running_loss += loss.item()
            running_psnr += psnr
            running_ssim += ssim

            loop.set_postfix(loss=loss.item(), psnr=psnr, ssim=ssim)

            # Capture sample flow for the tracked image
            if sample_flow is None and config.data_dir in img_path:
                sample_flow = (masked_image[0], output[0], original_image[0])

        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_ssim = running_ssim / len(train_loader)

        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        train_ssims.append(avg_train_ssim)

        print(f"📊 Epoch {epoch+1}/{config.pretrain_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | PSNR: {avg_train_psnr:.2f} | SSIM: {avg_train_ssim:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = config._timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"pretrained_unet_epoch{epoch+1}_{timestamp}.pth"
            save_checkpoint(model, config.checkpoint_dir, filename)

    # Save final model
    timestamp = config._timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_filename = f"pretrained_unet_final_{timestamp}.pth"
    save_checkpoint(model, config.checkpoint_dir, final_filename)
    print(f"✅ Saved final pretrained model to {final_filename}")

    # Visualize sample flow
    if sample_flow:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_flow[0].cpu().squeeze(), cmap='gray')
        plt.title("Masked Input")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sample_flow[1].cpu().squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(sample_flow[2].cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(config.output_dir, "sample_flow_pretrain.png")
        plt.savefig(save_path)
        plt.show()
        print(f"📸 Saved pretraining sample flow to {save_path}")

    # Plot metrics
    plot_metrics(train_losses, train_psnrs, train_ssims, config.output_dir, "pretrain")

def plot_metrics(losses, psnrs, ssims, output_dir, phase):
    epochs = range(1, len(losses)+1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label="Train Loss", marker='o')
    plt.title(f"MSE Loss per Epoch ({phase})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, psnrs, label="Train PSNR", marker='o')
    plt.title(f"PSNR per Epoch ({phase})")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssims, label="Train SSIM", marker='o')
    plt.title(f"SSIM per Epoch ({phase})")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{phase}_metrics.png")
    plt.savefig(save_path)
    plt.show()
    print(f"📸 Saved {phase} metrics plot to {save_path}")

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

    train_loader, _, test_loader = get_dataloaders(config, mode='pretrain')
    model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)

    pretrain(model, train_loader, None, config)

if __name__ == "__main__":
    main()