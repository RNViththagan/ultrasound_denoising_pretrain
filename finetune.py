import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, save_checkpoint
from model import get_model
from datetime import datetime
import os

def finetune(model, train_loader, val_loader, config):
    """
    Fine-tune the model for Noisier2Noise denoising.
    - Input: Z (doubly-noisy, Y + Y*M).
    - Target: Y (singly-noisy, BUSI images, pseudo-clean).
    - Goal: Learn f(Z) ≈ Y.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.finetune_lr)

    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []

    for epoch in range(config.finetune_epochs):
        model.train()
        running_loss = 0
        running_psnr = 0

        loop = tqdm(train_loader, desc=f"[Finetune Epoch {epoch+1}/{config.finetune_epochs}]")
        for doubly_noisy, singly_noisy in loop:
            doubly_noisy, singly_noisy = doubly_noisy.to(config.device), singly_noisy.to(config.device)

            output = model(doubly_noisy)  # f(Z)
            loss = loss_fn(output, singly_noisy)  # MSE(f(Z), Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(loss).item()
            running_loss += loss.item()
            running_psnr += psnr

            loop.set_postfix(loss=loss.item(), psnr=psnr)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = 0
        with torch.no_grad():
            for doubly_noisy, singly_noisy in val_loader:
                doubly_noisy, singly_noisy = doubly_noisy.to(config.device), singly_noisy.to(config.device)
                val_output = model(doubly_noisy)
                v_loss = loss_fn(val_output, singly_noisy)
                val_loss += v_loss.item()
                val_psnr += calculate_psnr(v_loss).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        # Logging
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)

        print(f"📊 Epoch {epoch+1}/{config.finetune_epochs} | Train Loss: {avg_train_loss:.4f} | PSNR: {avg_train_psnr:.2f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"finetuned_resnet_noise{config.noise_std}_epoch{epoch+1}_{timestamp}.pth"
            save_checkpoint(model, config.checkpoint_dir, filename)

    # Save final fine-tuned model
    config._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_filename = f"finetuned_resnet_noise{config.noise_std}_final_{config._timestamp}.pth"
    save_checkpoint(model, config.checkpoint_dir, final_filename)

    # Plotting
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, config.noise_std, config.output_dir)

def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, noise_std, output_dir):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title(f"MSE Loss per Epoch (Noise Std={noise_std})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR", marker='o')
    plt.plot(epochs, val_psnrs, label="Val PSNR", marker='o')
    plt.title(f"PSNR per Epoch (Noise Std={noise_std})")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"finetune_metrics_noise{noise_std}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"📸 Saved metrics plot to {save_path}")

def main():
    from config import Config
    from dataset import get_dataloaders
    from utils import seed_everything, print_gpu_info

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    seed_everything(42)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("./outs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    config.output_dir = output_dir  # Store output_dir in config for plot_metrics
    train_loader, val_loader, test_loader = get_dataloaders(config, mode='finetune')

    model = get_model(model_name="resnet", pretrained=False).to(config.device)
    pretrained_path = os.path.join(config.checkpoint_dir, "pretrained_masked_unet_final.pth")
    model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
    print(f"✅ Loaded pretrained weights from {pretrained_path}")

    finetune(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()