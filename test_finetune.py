import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_psnr, calculate_ssim
from model import get_model
from datetime import datetime
import os

def test_finetune(model, test_loader, config):
    """
    Test the finetuned Noisier2Noise model.
    - Input: Z (doubly-noisy, Y + Y*M).
    - Target: Y (input, pseudo-clean).
    """
    model.eval()
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    sample_images = []

    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing Finetuned")
        for doubly_noisy, input in loop:
            doubly_noisy, input = doubly_noisy.to(config.device), input.to(config.device)

            output = model(doubly_noisy)  # f(Z)
            loss = loss_fn(output, input)
            test_loss += loss.item()
            test_psnr += calculate_psnr(loss).item()
            test_ssim += calculate_ssim(output, input).item()

            if len(sample_images) < 4:
                sample_images.append((doubly_noisy[0], output[0], input[0]))

    avg_test_loss = test_loss / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)

    print(f"📊 Finetuned Test Results (Input Y, Noise Std={config.noise_std}):")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test PSNR: {avg_test_psnr:.2f} dB")
    print(f"Average Test SSIM: {avg_test_ssim:.4f}")

    # Visualize sample images
    plt.figure(figsize=(12, 8))
    for i, (doubly_noisy, output, input) in enumerate(sample_images):
        plt.subplot(4, 3, i*3 + 1)
        plt.imshow(doubly_noisy.cpu().squeeze(), cmap='gray')
        plt.title("Doubly-Noisy")
        plt.axis('off')

        plt.subplot(4, 3, i*3 + 2)
        plt.imshow(output.cpu().squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

        plt.subplot(4, 3, i*3 + 3)
        plt.imshow(input.cpu().squeeze(), cmap='gray')
        plt.title("Input")
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(config.output_dir, f"test_results_finetune_noise{config.noise_std}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"📸 Saved visualization to {save_path}")

    # Save as output.png for consistency
    output_path = os.path.join(config.output_dir, "output.png")
    plt.savefig(output_path)
    print(f"📸 Saved visualization as {output_path}")

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

    _, _, test_loader = get_dataloaders(config, mode='finetune')
    model = get_model(model_name="resnet", pretrained=False).to(config.device)

    checkpoint_path = os.path.join(config.checkpoint_dir, f"finetuned_resnet_noise{config.noise_std}_final.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"🧪 Testing on {len(test_loader.dataset)} test images")
    test_finetune(model, test_loader, config)

if __name__ == "__main__":
    main()