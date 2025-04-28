import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from utils import calculate_psnr, save_checkpoint

def pretrain(model, train_loader, val_loader, config, pretrained_path=None):
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
    else:
        print("⚠️ No pretrained weights provided. Starting from scratch.")

    def masked_mse_loss(output, target, mask):
        """MSE loss only on unmasked pixels."""
        return ((output - target) ** 2 * mask).sum() / mask.sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []

    for epoch in range(config.pretrain_epochs):
        model.train()
        running_loss = 0
        running_psnr = 0

        loop = tqdm(train_loader, desc=f"[Pretrain Epoch {epoch+1}/{config.pretrain_epochs}]")
        for masked_batch, original_batch, mask in loop:
            masked_batch, original_batch, mask = masked_batch.to(config.device), original_batch.to(config.device), mask.to(config.device)

            output = model(masked_batch)
            loss = masked_mse_loss(output, original_batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Approximate PSNR for logging (on unmasked pixels)
            mse = ((output - original_batch) ** 2 * mask).sum() / mask.sum()
            psnr = calculate_psnr(mse).item()
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
            for masked_batch, original_batch, mask in val_loader:
                masked_batch, original_batch, mask = masked_batch.to(config.device), original_batch.to(config.device), mask.to(config.device)
                output = model(masked_batch)
                v_loss = masked_mse_loss(output, original_batch, mask)
                val_loss += v_loss.item()
                mse = ((output - original_batch) ** 2 * mask).sum() / mask.sum()
                val_psnr += calculate_psnr(mse).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        # Logging
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)

        print(f"📊 Epoch {epoch+1}/{config.pretrain_epochs} | Train Loss: {avg_train_loss:.4f} | PSNR: {avg_train_psnr:.2f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, config.checkpoint_dir, f"pretrained_masked_unet_epoch_{epoch+1}.pth")

    # Save final pretrained model
    save_checkpoint(model, config.checkpoint_dir, "pretrained_masked_unet_final.pth")

    # Plotting
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs)

def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("Masked MSE Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR", marker='o')
    plt.plot(epochs, val_psnrs, label="Val PSNR", marker='o')
    plt.title("PSNR per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    from config import Config
    from dataset import get_dataloaders
    from model import get_model
    from utils import seed_everything, print_gpu_info

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    seed_everything(42)

    config = Config()
    train_loader, val_loader, _ = get_dataloaders(config, mode='pretrain')

    model = get_model(model_name="resnet", pretrained=True).to(config.device)
    pretrain(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()