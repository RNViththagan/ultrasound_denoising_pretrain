import argparse
import os
import torch
from datetime import datetime
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
    seed_everything(config.random_seed)
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
        print(f"\n🚀 Starting Pretraining (Noise2Void) for {model_name.upper()}...")
        pretrain(model, train_loader_pretrain, None, config)
        test_pretrain(model, test_loader_pretrain, config)

        # Finetune (Noisier2Noise)
        print(f"\n🚀 Starting Fine-tuning (Noisier2Noise) for {model_name.upper()}...")
        pretrained_checkpoint = os.path.join(config.checkpoint_dir, f"pretrained_unet_final_{timestamp}.pth" if model_name == 'unet' else f"pretrained_resnet_final_{timestamp}.pth")
        if os.path.exists(pretrained_checkpoint):
            model.load_state_dict(torch.load(pretrained_checkpoint, map_location=config.device))
            print(f"✅ Loaded pretrained weights from {pretrained_checkpoint}")
        else:
            print(f"⚠️ Pretrained checkpoint not found at {pretrained_checkpoint}. Using {'ImageNet' if model_name == 'resnet' else 'current'} weights.")
        finetune(model, train_loader_finetune, None, config, skip_pretrain=(model_name == 'resnet'))
        test_finetune(model, test_loader_finetune, config, config.num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Image Denoising Pipeline")
    parser.add_argument('--model', type=str, choices=['unet', 'resnet', 'both'], default='both',
                        help="Model to run: 'unet' (MedSegUNet), 'resnet' (ModifiedResNet), or 'both'")
    args = parser.parse_args()
    main(args)