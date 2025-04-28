import os
from PIL import Image
import torch
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm

def create_augmented_dataset(input_dir, output_dir, num_augmentations=5):
    """
    Generate an augmented BUSI dataset by creating multiple augmented versions of each image.
    Save to output_dir with the same structure (benign/, malignant/, normal/).
    """
    # Define augmentation pipeline
    augmentation_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.1), ratio=(1.0, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    # Create output directories
    for label in ['benign', 'malignant', 'normal']:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    total_images = 0
    class_counts = {'benign': 0, 'malignant': 0, 'normal': 0}

    print(f"🕒 Augmentation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")

    # Process each class folder
    for label in ['benign', 'malignant', 'normal']:
        input_folder = os.path.join(input_dir, label)
        output_folder = os.path.join(output_dir, label)

        if not os.path.exists(input_folder):
            print(f"Warning: Folder {input_folder} does not exist.")
            continue

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📷 Processing {len(image_files)} images in {label}...")

        for image_file in tqdm(image_files, desc=f"Augmenting {label}"):
            image_path = os.path.join(input_folder, image_file)
            try:
                image = Image.open(image_path).convert('L')
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

            # Save original image
            original_save_path = os.path.join(output_folder, image_file)
            image.save(original_save_path)
            class_counts[label] += 1
            total_images += 1

            # Generate and save augmented images
            base_name, ext = os.path.splitext(image_file)
            for i in range(num_augmentations):
                # Apply augmentations
                augmented_image = augmentation_pipeline(image)
                # Convert back to PIL Image for saving
                if isinstance(augmented_image, torch.Tensor):
                    augmented_image = transforms.ToPILImage()(augmented_image)
                # Save augmented image
                aug_save_path = os.path.join(output_folder, f"{base_name}_aug_{i+1}{ext}")
                augmented_image.save(aug_save_path)
                class_counts[label] += 1
                total_images += 1

    print(f"✅ Augmentation complete!")
    print(f"Total Images Generated: {total_images}")
    print(f"Class Counts: {class_counts}")

def main():
    input_dir = "../Data_sets/BUSI/"
    output_dir = "../Data_sets/BUSI_augmented/"
    num_augmentations = 5  # Number of augmented versions per image

    create_augmented_dataset(input_dir, output_dir, num_augmentations)

if __name__ == "__main__":
    main()