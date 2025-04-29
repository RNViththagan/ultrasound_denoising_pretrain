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
    Pretrain the model using Noise2Void (masked reconstruction).
    - Input: Masked image (X * M).
    - Target: Original image (X), loss computed on unmasked pixels.
    - Goal: Learn f(X * M) ≈ X.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)

    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []

    # Early stopping
    best_metric = float('-inf') if config.early_stop_metric != "loss" else float('inf')
    best_model_state = None
    patience_counter = 0
    patience = config.early_stop_patience_pretrain
    min_delta = config.early_stop_min_delta
    maximize = config.early_stop_metric != "loss"  # Maximize SSIM/PSNR, minimize loss

    for epoch in range(config.pretrain_epochs):
        model.train()
        running_loss = 0
        running_psnr = 0
        running_ssim = 0

        loop = tqdm(train_loader, desc=f"[Pretrain Epoch {epoch+1}/{config.pretrain_epochs}]")
        for masked_image, original_image, mask in loop:
            masked_image = masked_image.to(config.device)
            original_image = original_image.to(config.device)
            mask = mask.to(config.device)

            output = model(masked_image)  # f(X * M)
            loss = loss_fn(output * mask, original_image * mask)  # MSE on unmasked pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = calculate_psnr(loss).item()
            ssim = calculate_ssim(output * mask, original_image * mask).item()
            running_loss += loss.item()
            running_psnr += psnr
            running_ssim += ssim

            loop.set_postfix(loss=loss.item(), psnr=psnr, ssim=ssim)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = running_psnr / len(train_loader)
        avg_train_ssim = running_ssim / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        with torch.no_grad():
            for masked_image, original_image, mask in val_loader:
                masked_image = masked_image.to(config.device)
                original_image = original_image.to(config.device)
                mask = mask.to(config.device)
                val_output = model(masked_image)
                v_loss = loss_fn(val_output * mask, original_image * mask)
                val_loss += v_loss.item()
                val_psnr += calculate_psnr(v_loss).item()
                val_ssim += calculate_ssim(val_output * mask, original_image * mask).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)

        # Logging
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)
        train_ssims.append(avg_train_ssim)
        val_ssims.append(avg_val_ssim)

        print(f"📊 Epoch {epoch+1}/{config.pretrain_epochs} | Train Loss: {avg_train_loss:.4f} | PSNR: {avg_train_psnr:.2f} | SSIM: {avg_train_ssim:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} | Val SSIM: {avg_val_ssim:.4f}")

        # Early stopping
        current_metric = {
            "ssim": avg_val_ssim,
            "psnr": avg_val_psnr,
            "loss": avg_val_loss
        }[config.early_stop_metric]

        improved = (current_metric > best_metric + min_delta) if maximize else (current_metric < best_metric - min_delta)
        if improved:
            best_metric = current_metric
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            best_filename = f"pretrained_masked_unet_best_{timestamp}.pth"
            save_checkpoint(model, config.checkpoint_dir, best_filename)
            print(f"✅ Saved best model (Epoch {epoch+1}, {config.early_stop_metric}: {best_metric:.4f}) to {best_filename}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement in {config.early_stop_metric}.")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"pretrained_masked_unet_epoch{epoch+1}_{timestamp}.pth"
            save_checkpoint(model, config.checkpoint_dir, filename)

    # Save final model with best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        config._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = f"pretrained_masked_unet_final_{config._timestamp}.pth"
        save_checkpoint(model, config.checkpoint_dir, final_filename)
        print(f"✅ Restored best weights and saved final model to {final_filename}")
    else:
        config._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = f"pretrained_masked_unet_final_{config._timestamp}.pth"
        save_checkpoint(model, config.checkpoint_dir, final_filename)

    # Plotting
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, train_ssims, val_ssims, config.output_dir)

def plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs, train_ssims, val_ssims, output_dir):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("MSE Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR", marker='o')
    plt.plot(epochs, val_psnrs, label="Val PSNR", marker='o')
    plt.title("PSNR per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_ssims, label="Train SSIM", marker='o')
    plt.plot(epochs, val_ssims, label="Val SSIM", marker='o')
    plt.title("SSIM per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "pretrain_metrics.png")
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

    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(config, mode='pretrain')
    model = get_model(model_name="resnet", pretrained=False).to(config.device)

    pretrain(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()