import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from monai.transforms import Compose, RandRotate, RandFlip, RandAdjustContrast, RandGaussianNoise, Rand2DElastic

class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, dataset_name, transform=None, mode='pretrain', mask_ratio=0.1, noise_std=0.1, split='train', exclude_paths=None, augment=False):
        self.root_dir = root_dir
        self.dataset_name = dataset_name.lower()  # 'busi' or 'hc18'
        self.transform = transform
        self.mode = mode  # 'pretrain' (Noise2Void), 'finetune' (Noisier2Noise)
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.split = split  # 'train' or 'test'
        self.exclude_paths = exclude_paths or []  # Paths to exclude
        self.augment = augment  # Apply runtime augmentations for training
        self.image_paths = []
        self.class_counts = {}  # For BUSI: {'benign': N, 'malignant': N, 'normal': N}; For HC18: {'hc18': N}
        self.sample_image_path = None  # Path to sample image for visualization flow

        # Define augmentation pipeline for training
        self.augmentation = None
        if augment:
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

        # Load images based on dataset
        if self.dataset_name == 'busi':
            folders = ['benign', 'malignant', 'normal']
            self.class_counts = {'benign': 0, 'malignant': 0, 'normal': 0}
            for label in folders:
                folder = os.path.join(root_dir, label)
                if not os.path.exists(folder):
                    print(f"Warning: Folder {folder} does not exist.")
                    continue
                for file in os.listdir(folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(folder, file)
                        if file_path in self.exclude_paths:
                            continue
                        self.image_paths.append(file_path)
                        self.class_counts[label] += 1
        elif self.dataset_name == 'hc18':
            folders = ['test_set', 'training_set']
            self.class_counts = {'hc18': 0}
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder {folder_path} does not exist.")
                    continue
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(folder_path, file)
                        if file_path in self.exclude_paths:
                            continue
                        self.image_paths.append(file_path)
                        self.class_counts['hc18'] += 1
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}. Supported: 'busi', 'hc18'.")

        if not self.image_paths:
            raise ValueError(f"No images found in the dataset directory for split={self.split}, dataset={self.dataset_name}.")

        # Select first image as sample for visualization flow
        if self.image_paths:
            self.sample_image_path = self.image_paths[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (256, 256), color=0)

        # Convert to numpy array and validate shape
        image_np = np.array(image)  # Shape: [H, W]
        if image_np.ndim != 2:
            print(f"Warning: Image {img_path} has unexpected shape {image_np.shape}. Creating default image.")
            image_np = np.zeros((256, 256), dtype=np.uint8)

        # Convert to tensor with channel dimension
        image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0) / 255.0  # Shape: [1, H, W]

        # Apply runtime augmentations for training
        if self.augment and self.augmentation:
            image = self.augmentation(image)

        # Manual resize using interpolate
        image = image.unsqueeze(0)  # Shape: [1, 1, H, W]
        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
        image = image.squeeze(0)  # Shape: [1, H, W]

        if self.mode == 'pretrain':
            # Noise2Void: masked input
            mask = torch.ones_like(image)
            mask_indices = torch.randperm(image.numel())[:int(image.numel() * self.mask_ratio)]
            mask.view(-1)[mask_indices] = 0
            masked_image = image * mask
            return masked_image, image, mask, img_path
        else:
            # Noisier2Noise: doubly-noisy input (Z = Y + Y*M)
            noise = torch.randn_like(image) * self.noise_std
            input = image  # Y (pseudo-clean)
            doubly_noisy = input + input * noise  # Z = Y + Y*M
            doubly_noisy = torch.clamp(doubly_noisy, 0, 1)
            return doubly_noisy, input, img_path

    def get_stats(self):
        return {
            'total_images': len(self.image_paths),
            'class_counts': self.class_counts,
            'sample_image_path': self.sample_image_path
        }

def get_dataloaders(config, mode='pretrain'):
    # Load full dataset
    full_dataset = UltrasoundDataset(
        config.data_dir,
        config.dataset_name,
        config.transform,
        mode=mode,
        mask_ratio=0.1,
        noise_std=config.noise_std,
        split='full',
        augment=False
    )
    full_stats = full_dataset.get_stats()
    n_total = full_stats['total_images']
    n_train = int(n_total * config.split_ratio[0])  # 70% for train
    n_test = n_total - n_train  # 30% for test

    # Split into train and test
    train_set, test_set = random_split(full_dataset, [n_train, n_test])

    # Create train dataset with augmentations
    train_dataset = UltrasoundDataset(
        config.data_dir,
        config.dataset_name,
        config.transform,
        mode=mode,
        mask_ratio=0.1,
        noise_std=config.noise_std,
        split='train',
        exclude_paths=[full_dataset.image_paths[i] for i in test_set.indices],
        augment=True  # Re-enable augmentations
    )

    print(f"📊 Dataset Statistics ({config.dataset_name}):")
    print(f"Total Images (Train): {len(train_set)}")
    print(f"Total Images (Test): {len(test_set)}")
    print(f"Train Class Counts: {train_dataset.get_stats()['class_counts']}")
    print(f"Test Class Counts: {full_stats['class_counts']}")
    print(f"Sample Image Path: {full_stats['sample_image_path']}")
    print(f"Train/Test Split: {len(train_set)}/{len(test_set)} "
          f"({config.split_ratio[0]*100:.1f}%/{config.split_ratio[2]*100:.1f}%)")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    sample_batch = next(iter(train_loader))
    if mode == 'pretrain':
        print(f"Sample Batch Shapes: Masked={sample_batch[0].shape}, Original={sample_batch[1].shape}, Mask={sample_batch[2].shape}")
    else:
        print(f"Sample Batch Shapes: Doubly-Noisy={sample_batch[0].shape}, Input={sample_batch[1].shape}")

    return train_loader, None, test_loader