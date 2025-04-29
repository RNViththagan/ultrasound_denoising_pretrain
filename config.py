import torch
import os

class Config:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset
        self.data_dir = "../Data_sets/BUSI_augmented/"
        self.split_ratio = [0.7, 0.15, 0.15]  # train/val/test
        self.batch_size = 8

        # Model
        self.model_name = "resnet"

        # Training
        self.pretrain_epochs = 150
        self.finetune_epochs = 100
        self.pretrain_lr = 1e-4
        self.finetune_lr = 1e-5

        # Early Stopping
        self.early_stop_patience_pretrain = 20
        self.early_stop_patience_finetune = 15
        self.early_stop_min_delta = 0.001
        self.early_stop_metric = "ssim"  # Options: "ssim", "psnr", "loss"

        # Noise/Mask Parameters
        self.noise_std = 0.1
        self.mask_ratio = 0.1

        # Checkpoint and Output
        self.checkpoint_dir = "checkpoints/"
        self.output_dir = "./outs/"  # Updated in main.py
        self._timestamp = None  # Set during training

        # Transform
        self.transform = None  # Add transforms if needed

    def __str__(self):
        return (f"Config:\n"
                f"  Device: {self.device}\n"
                f"  Data Dir: {self.data_dir}\n"
                f"  Split Ratio: {self.split_ratio}\n"
                f"  Batch Size: {self.batch_size}\n"
                f"  Model: {self.model_name}\n"
                f"  Pretrain Epochs: {self.pretrain_epochs}\n"
                f"  Finetune Epochs: {self.finetune_epochs}\n"
                f"  Pretrain LR: {self.pretrain_lr}\n"
                f"  Finetune LR: {self.finetune_lr}\n"
                f"  Early Stop Patience (Pretrain): {self.early_stop_patience_pretrain}\n"
                f"  Early Stop Patience (Finetune): {self.early_stop_patience_finetune}\n"
                f"  Early Stop Min Delta: {self.early_stop_min_delta}\n"
                f"  Early Stop Metric: {self.early_stop_metric}\n"
                f"  Noise Std: {self.noise_std}\n"
                f"  Mask Ratio: {self.mask_ratio}\n"
                f"  Checkpoint Dir: {self.checkpoint_dir}\n"
                f"  Output Dir: {self.output_dir}")