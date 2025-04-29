import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import random

def create_augmented_dataset(input_dir, output_dir, num_augmentations=5):
    """
    Create an augmented dataset from the BUSI dataset.
    For each image, save the original and generate augmented versions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in ['benign', 'malignant', 'normal']:
        input_folder = os.path.join(input_dir, label)
        output_folder = os.path.join(output_dir, label)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"Warning: Input folder {input_folder} does not exist.")
            continue

        for file in os.listdir(input_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load original image
                img_path = os.path.join(input_folder, file)
                img = Image.open(img_path).convert('L')

                # Save original image
                original_output_path = os.path.join(output_folder, file)
                img.save(original_output_path)

                # Define augmentation transforms
                augmentations = [
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomResizedCrop(size=img.size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ]

                # Generate augmented images
                for i in range(num_augmentations):
                    aug_img = img.copy()
                    # Apply a random subset of augmentations
                    random.shuffle(augmentations)
                    for aug in augmentations[:random.randint(1, len(augmentations))]:
                        aug_img = aug(aug_img)
                    aug_output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_aug_{i+1}.png")
                    aug_img.save(aug_output_path)

                print(f"Processed {file}: Saved original and {num_augmentations} augmented images to {output_folder}")

def main(num_augmentations=5):
    input_dir = "../Data_sets/BUSI/"
    output_dir = "../Data_sets/BUSI_augmented/"
    create_augmented_dataset(input_dir, output_dir, num_augmentations)
    print(f"✅ Augmentation complete. Dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create augmented BUSI dataset")
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help="Number of augmentations per image (default: 5)")
    args = parser.parse_args()
    main(num_augmentations=args.num_augmentations)