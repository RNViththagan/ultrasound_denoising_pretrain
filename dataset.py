import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np

class BUSIDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='pretrain', mask_ratio=0.1, noise_std=0.1):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  # 'pretrain' (Noise2Void), 'finetune' (Noisier2Noise)
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.image_paths = []
        self.class_counts = {'benign': 0, 'malignant': 0, 'normal': 0}

        for label in ['benign', 'malignant', 'normal']:
            folder = os.path.join(root_dir, label)
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist.")
                continue
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder, file))
                    self.class_counts[label] += 1

        if not self.image_paths:
            raise ValueError("No images found in the dataset directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (256, 256), color=0)

        if self.transform:
            image = self.transform(image)

        if self.mode == 'pretrain':
            # Noise2Void: masked input
            mask = torch.ones_like(image)
            mask_indices = torch.randperm(image.numel())[:int(image.numel() * self.mask_ratio)]
            mask.view(-1)[mask_indices] = 0
            masked_image = image * mask
            return masked_image, image, mask
        else:
            # Noisier2Noise: doubly-noisy input (Z = Y + Y*M), singly-noisy target (Y)
            noise = torch.randn_like(image) * self.noise_std
            singly_noisy = image  # Y (BUSI image, assumed singly-noisy)
            doubly_noisy = singly_noisy + singly_noisy * noise  # Z = Y + Y*M
            doubly_noisy = torch.clamp(doubly_noisy, 0, 1)
            return doubly_noisy, singly_noisy  # Z, Y

    def get_stats(self):
        return {
            'total_images': len(self.image_paths),
            'class_counts': self.class_counts
        }

def get_dataloaders(config, mode='pretrain'):
    dataset = BUSIDataset(config.data_dir, config.transform, mode=mode, mask_ratio=0.1, noise_std=config.noise_std)
    stats = dataset.get_stats()
    print("📊 Dataset Statistics:")
    print(f"Total Images: {stats['total_images']}")
    print(f"Class Counts: {stats['class_counts']}")

    n_total = len(dataset)
    n_train = int(n_total * config.split_ratio[0])
    n_val = int(n_total * config.split_ratio[1])
    n_test = n_total - n_train - n_val

    print(f"Train/Val/Test Split: {n_train}/{n_val}/{n_test} "
          f"({config.split_ratio[0]*100:.1f}%/{config.split_ratio[1]*100:.1f}%/{config.split_ratio[2]*100:.1f}%)")

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    sample_batch = next(iter(train_loader))
    if mode == 'pretrain':
        print(f"Sample Batch Shapes: Masked={sample_batch[0].shape}, Original={sample_batch[1].shape}, Mask={sample_batch[2].shape}")
    else:
        print(f"Sample Batch Shapes: Doubly-Noisy={sample_batch[0].shape}, Singly-Noisy={sample_batch[1].shape}")

    return train_loader, val_loader, test_loader