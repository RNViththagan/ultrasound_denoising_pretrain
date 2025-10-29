import argparse
import os
import torch
from datetime import datetime
import random
from train import pretrain
from finetune import finetune
from test_pretrain import test_pretrain
from test_finetune import test_finetune
from config import Config
from dataset import get_dataloaders
from model import get_model
from utils import seed_everything, print_gpu_info

def main(args):
    """
    Ultrasound Image Denoising Pipeline

    This script orchestrates the denoising pipeline for BUSI or HC18 datasets using a MedSeg U-Net.
    It supports three modes:
    1. pretrain_finetune: Pretrain with Noise2Void (N2V), save model, fine-tune with Noisier2Noise (N2N), save model, test both.
    2. finetune_pretrained: Fine-tune using a saved pretrained model, test fine-tuned model.
    3. finetune_scratch: Fine-tune from random weights, test fine-tuned model.

    Run Instructions:
    1. Install dependencies:
       pip install monai torch torchvision numpy matplotlib tqdm Pillow pytorch_ssim
    2. Prepare datasets:
       - BUSI: Place at ../Data_sets/BUSI/ with subfolders benign/, malignant/, normal/
       - HC18: Place at ../Data_sets/HC18/ with subfolders test_set/, training_set/
    3. Configure settings in config.py:
       - Set dataset_name to "BUSI" or "HC18"
       - Adjust batch_size, image_size, epochs, noise_std, mask_ratio as needed
    4. Run the pipeline:
       - Mode 1: python main.py --mode pretrain_finetune
       - Mode 2: python main.py --mode finetune_pretrained --pretrained_path checkpoints/pretrained_unet_final_<timestamp>.pth
       - Mode 3: python main.py --mode finetune_scratch
    5. Outputs:
       - Checkpoints: checkpoints/pretrained_unet_final_*.pth, finetuned_unet_noise0.1_final_*.pth
       - Visualizations: outs/<timestamp>/sample_flow_*.png, sample_images_*.png, metrics_*.png
       - Metrics: Loss, PSNR, SSIM displayed per epoch and for tests

    Notes:
    - Ensure GPU is available for faster training (CUDA support in config.py).
    - Sample images track one test image through all phases (pretrain, finetune, test).
    - Metrics plots show loss, PSNR, SSIM vs. epochs.
    """
    print(f"🕒 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_gpu_info()

    # Initialize config
    config = Config()
    if args.noise_std is not None:
        config.noise_std = args.noise_std
    if args.finetune_epochs is not None:
        config.finetune_epochs = args.finetune_epochs
    if args.pretrain_epochs is not None:
        config.pretrain_epochs = args.pretrain_epochs
    if args.dataset.lower() == 'hc18':
        config.dataset_name = "HC18"
        config.data_dir = f"../Data_sets/HC18/"
    if args.dataset.lower() == 'busi':
        config.dataset_name = "BUSI"
        config.data_dir = f"../Data_sets/BUSI/"
    if args.seed is not None:
        config.random_seed = args.seed
    else:
        config.random_seed = random.randint(0, 100000)
    seed_everything(config.random_seed)
    print(f"🌱 Using random seed: {config.random_seed}")

    dataset_dir = config.data_dir
    if not os.path.exists(dataset_dir) or not any(os.listdir(dataset_dir)):
        raise ValueError(f"Dataset not found at {dataset_dir}. Ensure {config.dataset_name} dataset is available.")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.output_dir = os.path.join("./outs", timestamp)
    os.makedirs(config.output_dir, exist_ok=True)
    config._timestamp = timestamp

    # Get data loaders
    train_loader_pretrain, _, test_loader_pretrain = get_dataloaders(config, mode='pretrain')
    train_loader_finetune, _, test_loader_finetune = get_dataloaders(config, mode='finetune')

    models = ['unet'] if args.model == 'unet' else ['resnet'] if args.model == 'resnet' else ['unet', 'resnet']
    
    for model_name in models:
        print(f"\n🚀 Processing model: {model_name.upper()}")
        config.output_dir = os.path.join("./outs", timestamp, model_name)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        model = get_model(model_name=model_name, pretrained=(model_name == 'resnet')).to(config.device)
        
        # Pretrain (Noise2Void)
        if not args.skip_pretrain:
            print(f"\n🚀 Starting Pretraining (Noise2Void) for {model_name.upper()}...")
            pretrain(model, train_loader_pretrain, None, config)
            test_pretrain(model, test_loader_pretrain, config)

        # Finetune (Noisier2Noise)
        print(f"\n🚀 Starting Fine-tuning (Noisier2Noise) for {model_name.upper()}...")
        pretrained_checkpoint = os.path.join(config.checkpoint_dir, f"pretrained_unet_final_{timestamp}.pth" if model_name == 'unet' else f"pretrained_resnet_final_{timestamp}.pth")
        if os.path.exists(pretrained_checkpoint) and not args.skip_pretrain and args.use_checkpoint:
            model.load_state_dict(torch.load(pretrained_checkpoint, map_location=config.device))
            print(f"✅ Loaded pretrained weights from {pretrained_checkpoint}")
        else:
            print(f"⚠️ Pretrained checkpoint not found or skipped. Using {'ImageNet' if model_name == 'resnet' else 'random'} weights.")
        finetune(model, train_loader_finetune, None, config, skip_pretrain=(model_name == 'resnet' or args.skip_pretrain))
        test_finetune(model, test_loader_finetune, config, config.num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Image Denoising Pipeline")
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'both'], default='both',
                        help="Model to run: 'unet' (MedSegUNet), 'resnet' (ModifiedResNet), or 'both'")
    parser.add_argument('--noise_std', type=float, default=None,
                        help="Standard deviation of noise for training (overrides config.noise_std if provided)")
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help="Skip pretraining and proceed directly to fine-tuning")
    parser.add_argument('--finetune_epochs', type=int, default=None,
                        help="Number of fine-tuning epochs (overrides config.finetune_epochs if provided)")
    parser.add_argument('--pretrain_epochs', type=int, default=None,
                        help="Number of pre-training epochs (overrides config.pretrain_epochs if provided)")
    parser.add_argument('--dataset', type=str, default='busi', choices=['busi', 'hc18'],
                        help="Dataset to use for training (default: 'busi')")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for training (overrides config.random_seed if provided; random if not set)")
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                        help="Use existing pretrained checkpoint if available")
    args = parser.parse_args()
    main(args)