#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Define Dataset Class for CSV Files
Multi-Label Classification Dataset for Equipment Activities
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torchvision.models as models
import glob
import numpy as np
from torchvision.models import ResNet18_Weights
import random

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define a mapping from activity strings to integers
activity_mapping = {
    'Idle': 0,
    'Swing Bucket': 1,
    'Load Bucket': 2,
    'Dump': 3,
    'Move': 4
}

# Dataset class for handling image data and activities
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

        if image is None:
            raise ValueError(f"Error loading image {frame_path}")

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        activities = eval(self.data.iloc[idx]["activities"])
        #print("Activities:", activities)  # This will show what activities are in the dataset

        # Convert activity strings to binary multi-label format
        #activities = eval(self.data.iloc[idx]["activities"])
        #print("Activities:", activities)  # This will show what activities are in the dataset
        binary_activities = torch.zeros(len(activity_mapping), dtype=torch.float32)  # Initialize a binary vector

        for activity_str in activities:
            if activity_str in activity_mapping:
               activity_int = activity_mapping[activity_str]
               binary_activities[activity_int] = 1.0  # Set the corresponding index to 1
           # Debugging: Print activities
           
        #print(f"Dataset Index: {idx}, Frame File: {frame_file}")
        #print(f"Original Activities: {activities}")
        #print(f"Binary Encoded Activities: {binary_activities}")
        return image, binary_activities

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(train_loader)
    # Verify datasets
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Define CNN model using ResNet18 for Multi-Label Classification
    class ActivityCNN(nn.Module):
        def __init__(self, num_classes=5):  # Number of possible activities
            super(ActivityCNN, self).__init__()
            # Use weights instead of 'pretrained=True'
            self.extra_conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Additional conv layer
                                            nn.BatchNorm2d(32),  # Batch normalization
                                            nn.ReLU(),  # Activation function
                                            nn.MaxPool2d(kernel_size=2, stride=2))  # Downsampling
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #Input Tensor: [16, 3, 224, 224] (16 RGB images of size 224x224).
            self.model.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Modify final layer Output Tensor: [16, 5] (Multi-label logits for 5 activities)
        
        def forward(self, x):
            
            # Pass through extra convolutional layers (added by us)
            x = self.extra_conv(x)
            # Pass the processed output through the ResNet18 model
            x = self.model(x)
            return x

# Initialize model, loss function, and optimizer
model = ActivityCNN(num_classes=len(activity_mapping))
criterion = nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Move model to GPU if available
if torch.cuda.is_available():
   model = model.cuda()
    
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    # Store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()  # Ensure labels are float

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Inside your training loop
	    #print("Model outputs (logits):", outputs)
            #print("Labels:", labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} finished with Train Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for images, labels in val_loader:
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.float().cuda()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Apply sigmoid to outputs and threshold at 0.5 for multi-label classification
                preds = torch.sigmoid(outputs) > 0.5

                # Multi-label accuracy (exact match count)
                correct += (preds == labels).sum().item()
                total += labels.numel()  # Total number of labels

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        save_path = "activity_cnn_model_multilabel_25.pth"
        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'epoch': num_epochs}, save_path)        
    print("Training complete.")

    # Plot the learning curve (Training and Validation Loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', marker='x')
    plt.title('Learning Curve (Training and Validation Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)



"""
4. Evaluate the Model
Once training is complete, evaluate the model on the test dataset.
"""
# Evaluate the model on the test set
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            # Apply sigmoid to outputs and threshold at 0.5 for multi-label classification
            preds = torch.sigmoid(outputs) > 0.5

            # Multi-label accuracy (exact match count)
            correct += (preds == labels).sum().item()
            total += labels.numel()  # Total number of labels

    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total  # Multi-label accuracy
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    return test_loss, test_acc

# After training, evaluate the model on the test set
test_loss, test_acc = evaluate_model(model, test_loader, criterion)

 
"""
4. Evaluate the Model
Once training is complete, evaluate the model on the test dataset:.
"""
'''
# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
            
            correct += (preds == labels.byte()).sum().item()
            total += labels.numel()
    
    test_acc = 100 * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')

# Start training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

# Evaluate the model
evaluate_model(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), "activity_cnn_model_multilabel.pth")
print("Model saved successfully.")
''' 
