import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim
from model import get_model
from datetime import datetime
import os

def test_pretrain(model, test_loader, config):
    """
    Test the pretrained Noise2Void model.
    - Input: Masked image (X * M).
    - Output: Reconstructed image.
    """
    model.eval()
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    sample_images = []

    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing Pretrained")
        for masked_image, original_image, mask in loop:
            masked_image = masked_image.to(config.device)
            original_image = original_image.to(config.device)
            mask = mask.to(config.device)

            output = model(masked_image)  # f(X * M)
            loss = loss_fn(output * mask, original_image * mask)
            test_loss += loss.item()
            test_psnr += calculate_psnr(loss).item()
            test_ssim += calculate_ssim(output * mask, original_image * mask).item()

            if len(sample_images) < 4:
                sample_images.append((masked_image[0], output[0], original_image[0]))

    avg_test_loss = test_loss / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)

    print(f"📊 Pretrained Test Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")

    # Visualize sample images
    plt.figure(figsize=(12, 4))
    for i, (masked, output, original) in enumerate(sample_images):
        plt.subplot(4, 3, i*3 + 1)
        plt.imshow(masked.cpu().squeeze(), cmap='gray')
        plt.title("Masked Input")
        plt.axis('off')

        plt.subplot(4, 3, i*3 + 2)
        plt.imshow(output.cpu().squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        plt.subplot(4, 3, i*3 + 3)
        plt.imshow(original.cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, "sample_images_pretrain.png")
    plt.savefig(save_path)
    plt.show()
    print(f"📸 Saved visualization to {save_path}")

def main():
    from config import Config
    from dataset import get_dataloaders
    from utils import seed_everything, print_gpu_info

    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    seed_everything(42)

    config = Config()
    # Create timestamped output directory for standalone execution
    if not config._timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.output_dir = os.path.join("./outs", timestamp)
        config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    _, _, test_loader = get_dataloaders(config, mode='pretrain')
    model = get_model(model_name="resnet", pretrained=False).to(config.device)

    checkpoint_path = os.path.join(config.checkpoint_dir, "pretrained_masked_unet_final.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"🧪 Testing on {len(test_loader.dataset)} test images")
    test_pretrain(model, test_loader, config)

if __name__ == "__main__":
    main()