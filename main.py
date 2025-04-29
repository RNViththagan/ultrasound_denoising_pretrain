import argparse
import os
from datetime import datetime
import augment_dataset
from train import pretrain
from finetune import finetune
from test_pretrain import test_pretrain
from test_finetune import test_finetune
from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import seed_everything, print_gpu_info


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

    # Check or create augmented dataset
    config = Config()
    augmented_dir = config.data_dir  # ../Data_sets/BUSI_augmented/
    if args.run_augmentation:
        print(f"📸 Running augmentation with num_augmentations={args.num_augmentations}")
        augment_dataset.main(num_augmentations=args.num_augmentations)
    elif not os.path.exists(augmented_dir) or not any(os.listdir(augmented_dir)):
        raise ValueError(f"Augmented dataset not found at {augmented_dir}. Run augmentation with --run_augmentation or create the dataset manually.")
    else:
        print(f"✅ Using existing augmented dataset at {augmented_dir}")

    # Create single timestamped output directory for all tasks
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    os.makedirs(config.output_dir, exist_ok=True)
    config._timestamp = timestamp  # Store for checkpoint naming

    # Get data loaders
    train_loader_pretrain, val_loader_pretrain, test_loader_pretrain = get_dataloaders(config, mode='pretrain')
    train_loader_finetune, val_loader_finetune, test_loader_finetune = get_dataloaders(config, mode='finetune')

    # Initialize model
    model = get_model(model_name="resnet", pretrained=True).to(config.device)

    if args.mode in ['pretrain', 'both']:
        print("\n🚀 Starting Pretraining (Noise2Void)...")
        train_loader, val_loader, test_loader = get_dataloaders(config, mode='pretrain')
        print("📸 Visualizing sample pretrain images...")
        visualize_sample_images(
            train_loader,
            mode='pretrain',
            save_path=os.path.join(config.output_dir, "sample_images_pretrain.png")
        )
        pretrain(model, train_loader_pretrain, val_loader_pretrain, config)
        test_pretrain(model, test_loader_pretrain, config)

    if args.mode in ['finetune', 'both']:
        print("\n🚀 Starting Fine-tuning (Noisier2Noise)...")
        train_loader, val_loader, test_loader = get_dataloaders(config, mode='finetune')
        print("📸 Visualizing sample finetune images...")
        visualize_sample_images(
            train_loader,
            mode='finetune',
            save_path=os.path.join(config.output_dir, "sample_images_finetune.png")
        )
        pretrained_path = os.path.join(config.checkpoint_dir, f"pretrained_masked_unet_final_{timestamp}.pth")
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}. Starting from scratch.")
        finetune(model, train_loader_finetune, val_loader_finetune, config)
        test_finetune(model, test_loader_finetune, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Image Denoising Pipeline")
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune', 'both'], default='both',
                        help="Mode to run: 'pretrain', 'finetune', or 'both'")
    parser.add_argument('--run_augmentation', action='store_true',
                        help="Run augmentation before training (default: use existing augmented dataset)")
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help="Number of augmentations per image (default: 5)")
    args = parser.parse_args()
    main(args)