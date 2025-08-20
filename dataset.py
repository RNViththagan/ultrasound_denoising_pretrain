import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from monai.transforms import Compose, RandRotate, RandFlip, RandAdjustContrast, RandGaussianNoise, Rand2DElastic

class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='pretrain',
                 mask_ratio=0.1, noise_std=0.1, split='train',
                 exclude_paths=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  # 'pretrain' (Noise2Void), 'finetune' (Noisier2Noise)
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.split = split  # 'train', 'val', or 'test'
        self.exclude_paths = exclude_paths or []  # Paths to exclude (e.g., test images for train/val)
        self.image_paths = []
        self.class_counts = {'benign': 0, 'malignant': 0, 'normal': 0}
        self.augment = augment

        # Runtime augmentation pipeline
        self.augmentation = None
        if self.augment:
            self.augmentation = Compose([
                RandRotate(range_x=15.0, prob=0.5),           # Rotate ±15 degrees
                RandFlip(spatial_axis=0, prob=0.5),           # Horizontal flip
                RandFlip(spatial_axis=1, prob=0.5),           # Vertical flip
                RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)), # Adjust contrast
                RandGaussianNoise(prob=0.2, std=0.05),        # Speckle-like Gaussian noise
                Rand2DElastic(
                    spacing=(30, 30),                         # Elastic deformation grid
                    magnitude_range=(1, 2),                   # Deformation magnitude
                    prob=0.3,
                    mode='bilinear'
                ),  # Elastic deformations
            ])

        # Collect BUSI images
        for label in ['benign', 'malignant', 'normal']:
            folder = os.path.join(root_dir, label)
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist.")
                continue
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder, file)
                    # For test split, include only original images (exclude *_aug_*.png)
                    if self.split == 'test' and '_aug_' in file:
                        continue
                    # Exclude paths used in other splits (e.g., test images from train/val)
                    if file_path in self.exclude_paths:
                        continue
                    self.image_paths.append(file_path)
                    self.class_counts[label] += 1

        if not self.image_paths:
            raise ValueError(f"No images found in dataset for split={self.split}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (256, 256), color=0)

        # Convert to tensor [1, H, W], normalized to [0,1]
        image_np = np.array(image)
        if image_np.ndim != 2:
            image_np = np.zeros((256, 256), dtype=np.uint8)
        # Convert to tensor with channel dimension
        image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0) / 255.0  # Shape: [1, H, W]

        # Apply runtime augmentation
        if self.augment and self.augmentation:
            image = self.augmentation(image)

        # Resize to [1, 256, 256]
        image = image.unsqueeze(0)  # Shape: [1, 1, H, W]
        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        image = image.squeeze(0)  # Shape: [1, H, W]

        if self.mode == 'pretrain':
            # Noise2Void: masked input
            mask = torch.ones_like(image)
            mask_indices = torch.randperm(image.numel())[:int(image.numel() * self.mask_ratio)]
            mask.view(-1)[mask_indices] = 0
            masked_image = image * mask
            return masked_image, image, mask
        else:
            # Noisier2Noise: doubly-noisy input (Z = Y + Y*M), input target (Y)
            noise = torch.randn_like(image) * self.noise_std
            input = image
            doubly_noisy = torch.clamp(input + input * noise, 0, 1)
            return doubly_noisy, input

    def get_stats(self):
        return {
            'total_images': len(self.image_paths),
            'class_counts': self.class_counts
        }

def get_dataloaders(config, mode='pretrain'):
    # Original dataset for test
    original_dataset = BUSIDataset(
        config.data_dir,
        config.transform,
        mode=mode,
        mask_ratio=0.1,
        noise_std=config.noise_std,
        split='test'
    )
    original_stats = original_dataset.get_stats()
    n_original = original_stats['total_images']  # ~780
    n_test = int(n_original * config.split_ratio[2])  # ~117 (15% of 780)

    # Load full dataset for train/val (original + augmented)
    full_dataset = BUSIDataset(
        config.data_dir,
        config.transform,
        mode=mode,
        mask_ratio=0.1,
        noise_std=config.noise_std,
        split='train',
        exclude_paths=original_dataset.image_paths,
        augment=True   # ✅ runtime augmentations
    )
    full_stats = full_dataset.get_stats()
    n_full = full_stats['total_images']  # ~4680 - n_original
    n_train = int(n_full * config.split_ratio[0] / (config.split_ratio[0] + config.split_ratio[1]))  # ~70% of (4680 - 780)
    n_val = n_full - n_train  # ~15% of (4680 - 780)

    # Split full_dataset into train and val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Split original_dataset for test
    test_set, _ = random_split(original_dataset, [n_test, n_original - n_test])

    print("📊 Dataset Statistics:")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    print(f"Train Class Counts: {full_stats['class_counts']}")
    print(f"Test Class Counts: {original_stats['class_counts']}")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    sample_batch = next(iter(train_loader))
    if mode == 'pretrain':
        print(f"Sample Batch Shapes: Masked={sample_batch[0].shape}, Original={sample_batch[1].shape}, Mask={sample_batch[2].shape}")
    else:
        print(f"Sample Batch Shapes: Doubly-Noisy={sample_batch[0].shape}, Input={sample_batch[1].shape}")

    return train_loader, val_loader, test_loader
