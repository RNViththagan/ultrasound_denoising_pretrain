import torch
from torchvision import transforms

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_epochs = 100  # Increased for robust pretraining
        self.finetune_epochs = 70
        self.pretrain_lr = 1e-4
        self.finetune_lr = 1e-5  # Lower for fine-tuning stability
        self.batch_size = 8
        self.data_dir = "../Data_sets/BUSI_augmented/"  # Use augmented dataset
        self.image_size = (256, 256)
        self.split_ratio = [0.7, 0.15, 0.15]  # train, val, test
        self.checkpoint_dir = "checkpoints/"
        self.noise_std = 0.1  # Speckle noise standard deviation for Noisier2Noise
        self.output_dir = "./outs"  # Default output directory, overridden by main.py or finetune.py

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])