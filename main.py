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
import glob

def main(args):
    """
    Ultrasound Image Denoising Pipeline

    This script orchestrates the denoising pipeline for BUSI or HC18 datasets using a MedSeg U-Net.
    It supports four modes:
    1. full_pipeline: Pretrain with Noise2Void (N2V), test pretrained model, fine-tune with Noisier2Noise (N2N) using latest pretrained checkpoint, test fine-tuned model.
    2. pretrain_test: Test the latest pretrained model from checkpoints/pretrained_unet_final_*.pth.
    3. finetune_latest: Fine-tune using the latest pretrained checkpoint, test fine-tuned model.
    4. finetune_test: Test the latest fine-tuned model from checkpoints/finetuned_unet_noise0.1_final_*.pth.

    Command Guide:
    1. Install Dependencies:
       ```bash
       pip install monai torch torchvision numpy matplotlib tqdm Pillow pytorch_ssim
       ```
    2. Prepare Datasets:
       - BUSI: Place at ../Data_sets/BUSI/ with subfolders benign/, malignant/, normal/ containing .png/.jpg images.
       - HC18: Place at ../Data_sets/HC18/ with subfolders test_set/, training_set/ containing .png/.jpg images.
       - Verify dataset path in config.py: `data_dir = '../Data_sets/<dataset_name>/'`.
    3. Configure Settings in config.py:
       - Set `dataset_name = 'BUSI'` or `'HC18'`.
       - Adjust `batch_size` (e.g., 8), `image_size` (e.g., (256, 256)), `noise_std` (e.g., 0.1), `finetune_epochs`, `random_seed`, etc.
       - Ensure `checkpoint_dir = './checkpoints/'` and `output_dir = './outs/'`.
    4. Run Commands:
       - Run full pipeline (pretrain, test pretrain, fine-tune with latest pretrained, test fine-tune):
         ```bash
         python main.py --mode full_pipeline
         ```
         - Output: Checkpoints in `checkpoints/pretrained_unet_final_*.pth`, `finetuned_unet_noise0.1_final_*.pth`; visualizations in `outs/<timestamp>/`.
       - Test latest pretrained model:
         ```bash
         python main.py --mode pretrain_test
         ```
         - Example checkpoint: `checkpoints/pretrained_unet_final_2025-08-06_01-36-07.pth`.
         - Output: Visualizations in `outs/<timestamp>/test_results_pretrain.png`.
       - Fine-tune with latest pretrained checkpoint and test:
         ```bash
         python main.py --mode finetune_latest
         ```
         - Automatically uses latest `checkpoints/pretrained_unet_final_*.pth`.
         - Output: Checkpoint in `checkpoints/finetuned_unet_noise0.1_final_*.pth`; visualizations in `outs/<timestamp>/`.
       - Test latest fine-tuned model:
         ```bash
         python main.py --mode finetune_test
         ```
         - Example checkpoint: `checkpoints/finetuned_unet_noise0.1_final_2025-08-06_01-36-07.pth`.
         - Output: Visualizations in `outs/<timestamp>/test_results_finetune_noise0.1.png`.
    5. Verify Outputs:
       - Checkpoints: `checkpoints/` for pretrained and fine-tuned models.
       - Visualizations: `outs/<timestamp>/sample_flow_*.png`, `test_results_*.png`, `metrics_*.png`.
       - Metrics: Console logs show loss (MSE + MSCN variance), PSNR, SSIM, BRISQUE per epoch and for tests.
    6. Troubleshooting:
       - Dataset not found: Ensure `../Data_sets/<dataset_name>/` exists with images.
       - Checkpoint not found: Run `full_pipeline` or `finetune_latest` to generate checkpoints.
       - Memory issues: Reduce `batch_size` in `config.py` (e.g., to 4) or use CPU (`device = 'cpu'`).
       - Training instability: Adjust `mscn_weight` in `finetune.py` and `test_finetune.py` (e.g., to 0.05).

    Notes:
    - Ensure GPU is available for faster training (CUDA support in config.py).
    - Sample images track one test image through all phases (pretrain, finetune, test).
    - Metrics plots show loss, PSNR, SSIM vs. epochs.
    - Uses latest pretrained checkpoint (`pretrained_unet_final_*.pth`) for fine-tuning and testing in relevant modes.
    - Uses latest fine-tuned checkpoint (`finetuned_unet_noise0.1_final_*.pth`) for testing in finetune_test mode.
    - Loss includes differentiable MSCN variance for perceptual quality optimization.
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
    config._timestamp = timestamp
    os.makedirs(config.output_dir, exist_ok=True)

    # Load data
    train_loader_pretrain, _, test_loader_pretrain = get_dataloaders(config, mode='pretrain')
    train_loader_finetune, _, test_loader_finetune = get_dataloaders(config, mode='finetune')

    # Select mode
    mode = args.mode

    if mode == 'full_pipeline':
        # Pretraining
        print("\n🚀 Starting Pretraining (Noise2Void)...")
        model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)
        pretrain(model, train_loader_pretrain, None, config)
        
        # Find and test the latest pretrained model
        pretrained_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "pretrained_unet_final_*.pth"))
        if not pretrained_checkpoints:
            raise FileNotFoundError(f"No pretrained checkpoints found in {config.checkpoint_dir}")
        pretrained_path = max(pretrained_checkpoints, key=os.path.getmtime)
        print(f"✅ Using latest pretrained checkpoint for testing: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
        test_pretrain(model, test_loader_pretrain, config)

        # Fine-tuning with the latest pretrained model
        print("\n🚀 Starting Fine-tuning (Noisier2Noise) with latest pretrained model...")
        model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
        finetune(model, train_loader_finetune, None, config, skip_pretrain=False)

        # Find and test the latest fine-tuned model
        finetuned_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "finetuned_unet_noise0.1_final_*.pth"))
        if not finetuned_checkpoints:
            raise FileNotFoundError(f"No fine-tuned checkpoints found in {config.checkpoint_dir}")
        finetuned_path = max(finetuned_checkpoints, key=os.path.getmtime)
        print(f"✅ Using latest fine-tuned checkpoint for testing: {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path, map_location=config.device))
        test_finetune(model, test_loader_finetune, config, config.num_samples)

    elif mode == 'pretrain_test':
        # Test the latest pretrained model
        pretrained_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "pretrained_unet_final_*.pth"))
        if not pretrained_checkpoints:
            raise FileNotFoundError(f"No pretrained checkpoints found in {config.checkpoint_dir}")
        pretrained_path = max(pretrained_checkpoints, key=os.path.getmtime)
        print(f"✅ Using latest pretrained checkpoint for testing: {pretrained_path}")
        model = get_model(model_name="unet", pretrained=True, pretrained_path=pretrained_path).to(config.device)
        test_pretrain(model, test_loader_pretrain, config, config.num_samples)

    elif mode == 'finetune_latest':
        # Fine-tune with the latest pretrained model
        pretrained_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "pretrained_unet_final_*.pth"))
        if not pretrained_checkpoints:
            raise FileNotFoundError(f"No pretrained checkpoints found in {config.checkpoint_dir}")
        pretrained_path = max(pretrained_checkpoints, key=os.path.getmtime)
        print("\n🚀 Starting Fine-tuning (Noisier2Noise) with latest pretrained model...")
        print(f"✅ Loaded pretrained weights from {pretrained_path}")
        model = get_model(model_name="unet", pretrained=True, pretrained_path=pretrained_path).to(config.device)
        finetune(model, train_loader_finetune, None, config, skip_pretrain=False)
        
        # Test the latest fine-tuned model
        finetuned_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "finetuned_unet_noise0.1_final_*.pth"))
        if not finetuned_checkpoints:
            raise FileNotFoundError(f"No fine-tuned checkpoints found in {config.checkpoint_dir}")
        finetuned_path = max(finetuned_checkpoints, key=os.path.getmtime)
        print(f"✅ Using latest fine-tuned checkpoint for testing: {finetuned_path}")
        model.load_state_dict(torch.load(finetuned_path, map_location=config.device))
        test_finetune(model, test_loader_finetune, config, config.num_samples)

    elif mode == 'finetune_test':
        # Test the latest fine-tuned model
        finetuned_checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "finetuned_unet_noise0.1_final_*.pth"))
        if not finetuned_checkpoints:
            raise FileNotFoundError(f"No fine-tuned checkpoints found in {config.checkpoint_dir}")
        finetuned_path = max(finetuned_checkpoints, key=os.path.getmtime)
        print(f"✅ Using latest fine-tuned checkpoint for testing: {finetuned_path}")
        model = get_model(model_name="unet", pretrained=False, pretrained_path=None).to(config.device)
        model.load_state_dict(torch.load(finetuned_path, map_location=config.device))
        test_finetune(model, test_loader_finetune, config, config.num_samples)

    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: full_pipeline, pretrain_test, finetune_latest, finetune_test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Image Denoising Pipeline")
    parser.add_argument('--mode', type=str, choices=['full_pipeline', 'pretrain_test', 'finetune_latest', 'finetune_test'],
                        default='full_pipeline', help="Mode: full_pipeline, pretrain_test, finetune_latest, or finetune_test")
    args = parser.parse_args()
    main(args)