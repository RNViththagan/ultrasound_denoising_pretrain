import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import calculate_psnr, print_gpu_info, seed_everything

def test_model(model, test_loader, config, checkpoint_path, save_dir="test_outputs_pretrain"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    print(f"✅ Loaded weights from {checkpoint_path}")

    model.eval()
    def masked_mse_loss(output, target, mask):
        return ((output - target) ** 2 * mask).sum() / mask.sum()

    test_loss = 0.0
    test_psnr = 0.0
    num_batches = len(test_loader)
    input_images, output_images, original_images = [], [], []

    with torch.no_grad():
        for i, (masked_batch, original_batch, mask) in enumerate(tqdm(test_loader, desc="Testing Pretrained")):
            masked_batch, original_batch, mask = masked_batch.to(config.device), original_batch.to(config.device), mask.to(config.device)
            output = model(masked_batch)
            loss = masked_mse_loss(output, original_batch, mask)
            test_loss += loss.item()
            psnr = calculate_psnr(loss).item()
            test_psnr += psnr

            if i == 0:
                input_images = masked_batch[:4].cpu()
                output_images = output[:4].cpu()
                original_images = original_batch[:4].cpu()

    avg_test_loss = test_loss / num_batches
    avg_test_psnr = test_psnr / num_batches
    print(f"📊 Pretrained Test Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")

    visualize_results(input_images, output_images, original_images, save_dir)

def visualize_results(input_images, output_images, original_images, save_dir):
    input_images = input_images * 0.5 + 0.5
    output_images = output_images * 0.5 + 0.5
    original_images = original_images * 0.5 + 0.5

    num_images = input_images.size(0)
    plt.figure(figsize=(12, 9))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(input_images[i].squeeze(), cmap='gray')
        plt.title("Masked Input")
        plt.axis('off')

        plt.subplot(3, num_images, i + num_images + 1)
        plt.imshow(output_images[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        plt.subplot(3, num_images, i + 2 * num_images + 1)
        plt.imshow(original_images[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_results_pretrain.png"))
    plt.show()
    print(f"📸 Saved visualization to {os.path.join(save_dir, 'test_results_pretrain.png')}")

def main():
    print_gpu_info()
    seed_everything(42)
    config = Config()
    _, _, test_loader = get_dataloaders(config, mode='pretrain')
    print(f"🧪 Testing on {len(test_loader.dataset)} test images")
    model = get_model(model_name="resnet", pretrained=False).to(config.device)
    checkpoint_path = os.path.join(config.checkpoint_dir, "pretrained_masked_unet_final.pth")
    test_model(model, test_loader, config, checkpoint_path)

if __name__ == "__main__":
    main()