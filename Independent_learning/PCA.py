import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
import torchvision.transforms as transforms
from PIL import Image


# Define Dataset class
class EquipmentActivityDataset(Dataset):
    def __init__(self, csv_file, frames_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def find_image_path(self, frame_file):
        search_pattern = os.path.join(self.frames_dir, '**', frame_file)
        matching_files = glob.glob(search_pattern, recursive=True)
        if len(matching_files) > 0:
            return matching_files[0]
        else:
            return None

    def __getitem__(self, idx):
        frame_file = self.data.iloc[idx]["frame"]
        frame_path = self.find_image_path(frame_file)

        if frame_path is None:
            raise ValueError(f"Error loading image {frame_file} - file not found in {self.frames_dir}")

        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if image is None:
            raise ValueError(f"Error loading image {frame_path}")

        # Convert to PIL image for transformation compatibility
        image = Image.fromarray(image)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        # Get activity labels for the sample
        activities = eval(self.data.iloc[idx]["activities"])

        return image, activities


# Define transformations for images (resize to a fixed size)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),         # Convert to tensor
])

if __name__ == "__main__":
    # Initialize the datasets
    # Initialize the datasets
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
    train_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/train_dataset.csv"
    val_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/val_dataset.csv"
    test_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/test_dataset.csv"

    train_dataset = EquipmentActivityDataset(csv_file=train_csv, frames_dir=frames_dir, transform=transform)
    val_dataset = EquipmentActivityDataset(csv_file=val_csv, frames_dir=frames_dir, transform=transform)
    test_dataset = EquipmentActivityDataset(csv_file=test_csv, frames_dir=frames_dir, transform=transform)

    # Create DataLoaders with reduced batch size to avoid memory issues
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=0)

    # Extracting features (pixel data) from the images
    image_features = []
    for images, _ in train_loader:
        images = images.numpy()
        # Flatten images to create 2D matrix: [num_samples, num_pixels]
        batch_size, channels, height, width = images.shape
        flat_images = images.reshape(batch_size, -1)  # Flattening each image
        image_features.append(flat_images)
    
    # Convert to a single numpy array
    image_features = np.vstack(image_features)

    # Normalize the features before PCA
    scaler = StandardScaler()
    image_features = scaler.fit_transform(image_features)

    # Perform PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    pca_result = pca.fit_transform(image_features)

    # Plot the PCA result
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolors='k', alpha=0.5)
    plt.title("PCA of Equipment Activity Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # Optionally, save the PCA model for future use
#    np.save("pca_model.npy", pca.components_)
#    print("PCA components saved to 'pca_model.npy'")
