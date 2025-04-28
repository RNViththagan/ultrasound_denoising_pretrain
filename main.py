import torch
from config import Config
from dataset import get_dataloaders
from model import get_model
from train import pretrain
from finetune import finetune
from test_pretrain import test_model as test_pretrain
from test_finetune import test_model as test_finetune
from utils import seed_everything, print_gpu_info
import os
import argparse
from datetime import datetime
from test_finetune import get_latest_checkpoint

def visualize_sample_images(loader, num_images=4, mode='pretrain', save_path="sample_images.png"):
    import matplotlib.pyplot as plt
    sample_batch = next(iter(loader))
    if mode == 'pretrain':
        masked_images, original_images, masks = sample_batch[:num_images]
        titles = ["Masked Input", "Original", "Mask"]
        images = [masked_images, original_images, masks]
    else:
        doubly_noisy, singly_noisy = sample_batch[:num_images]
        titles = ["Doubly-Noisy", "Singly-Noisy"]
        images = [doubly_noisy, singly_noisy]

    # Denormalize
    images = [img * 0.5 + 0.5 for img in images]

    plt.figure(figsize=(12, 4 * len(titles)))
    for i in range(num_images):
        for j, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(len(titles), num_images, i + 1 + j * num_images)
            plt.imshow(img[i].squeeze(), cmap='gray')
            plt.title(f"{title} {i+1}")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📸 Saved sample images to {save_path}")

def main(args):
    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()
    seed_everything(42)

    config = Config()
    print("🔧 Configuration:")
    print(f"Mode: {args.mode}")
    print(f"Device: {config.device}")
    print(f"Pretrain Epochs: {config.pretrain_epochs}")
    print(f"Finetune Epochs: {config.finetune_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.image_size}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Noise Std (Finetune): {config.noise_std}")

    pretrained_path = os.path.join(config.checkpoint_dir, "pretrained_masked_unet_final.pth")

    if args.mode in ['pretrain', 'both']:
        # Pretraining
        print("\n🚀 Starting Pretraining (Noise2Void)...")
        train_loader, val_loader, test_loader = get_dataloaders(config, mode='pretrain')
        print("📸 Visualizing sample pretrain images...")
        visualize_sample_images(train_loader, mode='pretrain', save_path="outs/sample_images_pretrain.png")

        model = get_model(model_name="resnet", pretrained=True).to(config.device)
        print("🧠 Model initialized: ModifiedResNet")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Check if pretrained model exists
        if os.path.exists(pretrained_path) and args.mode == 'pretrain':
            print(f"⚠️ Pretrained model found at {pretrained_path}. Loading instead of retraining.")
            model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
        else:
            pretrain(model, train_loader, val_loader, config, pretrained_path=None)

        # Test Pretrained Model
        print("\n🧪 Testing Pretrained Model...")
        test_pretrain(model, test_loader, config, checkpoint_path=pretrained_path)

    if args.mode in ['finetune', 'both']:
        # Fine-tuning
        print("\n🚀 Starting Fine-tuning (Noisier2Noise)...")
        train_loader, val_loader, test_loader = get_dataloaders(config, mode='finetune')
        print("📸 Visualizing sample finetune images...")
        visualize_sample_images(train_loader, mode='finetune', save_path="outs/sample_images_finetune.png")

        model = get_model(model_name="resnet", pretrained=False).to(config.device)
        print("🧠 Model initialized: ModifiedResNet")
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Load pretrained model
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}. Run pretraining first.")
        print(f"✅ Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=config.device))

        finetune(model, train_loader, val_loader, config)

        # Test Fine-tuned Model
        print("\n🧪 Testing Fine-tuned Model...")
        finetuned_path = get_latest_checkpoint(config.checkpoint_dir, config.noise_std)
        test_finetune(model, test_loader, config, checkpoint_path=finetuned_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pretraining, fine-tuning, or both for denoising.")
    parser.add_argument('--mode', type=str, default='both', choices=['pretrain', 'finetune', 'both'],
                        help="Mode to run: 'pretrain', 'finetune', or 'both'")
    args = parser.parse_args()
    main(args)