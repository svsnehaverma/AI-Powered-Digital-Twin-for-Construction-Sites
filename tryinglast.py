import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
import random
import numpy as np

# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define a mapping from activity strings to integers
activity_mapping = {
    'Idle': 0,
    'Swing Bucket': 1,
    'Load Bucket': 2,
    'Dump': 3,
    'Move': 4
}

# Define productivity contributions for each activity
activity_productivity = {
    0: 0.0,  # Idle
    1: 0.2,  # Swing Bucket
    2: 1.0,  # Load Bucket
    3: 0.8,  # Dump
    4: 0.6   # Move
}

# Define transformations for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VehicleActivityDataset(Dataset):
    def __init__(self, csv_file, frames_dir, transform=None, max_seq_len=10):
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.activity_mapping = activity_mapping

        # Debug activity distribution in the dataset
        activity_counts = self.data['activities'].apply(lambda x: len(eval(x))).value_counts()
        print(f"Activity counts in the dataset: {activity_counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_names = eval(self.data.iloc[idx, 0])  # List of image filenames
        activities = eval(self.data.iloc[idx, 2])  # List of activities
        bounding_boxes = eval(self.data.iloc[idx, 3])  # List of bounding boxes

        # Ensure bounding boxes are properly formatted
        bounding_boxes = [
            [bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']] 
            if isinstance(bbox, dict) else bbox for bbox in bounding_boxes
        ]

        # Map activities to integers and pad the sequence
        activity_seq = [self.activity_mapping[activity] for activity in activities]
        padded_activity_seq = activity_seq + [0] * (self.max_seq_len - len(activity_seq))
        padded_activity_seq = padded_activity_seq[:self.max_seq_len]

        # Use the most frequent valid activity as the target
        from collections import Counter
        valid_activity_counts = Counter(activity_seq)
        target = max(valid_activity_counts, key=valid_activity_counts.get) if valid_activity_counts else 0

        # Debug target selection
        #print(f"Row {idx}: Target activity is {target}")

        # Load images
        images = []
        for img_name in img_names:
            img_path = self._find_image_path(img_name)
            if img_path:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            else:
                print(f"Warning: Image {img_name} not found in {self.frames_dir}")

        # Pad with zero tensors if not enough images
        while len(images) < self.max_seq_len:
            images.append(torch.zeros(3, 224, 224))

        # Stack images and bounding boxes
        images = torch.stack(images[:self.max_seq_len])
        bbox_features = torch.tensor(bounding_boxes[:self.max_seq_len], dtype=torch.float32)

        return images, torch.tensor(target, dtype=torch.long), bbox_features

    def _find_image_path(self, img_name):
        for root, _, files in os.walk(self.frames_dir):
            if img_name in files:
                return os.path.join(root, img_name)
        return None

class VehicleActivityModel(nn.Module):
    def __init__(self, num_activities, seq_len):
        super(VehicleActivityModel, self).__init__()

        # ResNet feature extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove fully connected layer

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=16, dim_feedforward=2048, dropout=0.1, batch_first=True),
            num_layers=4
        )

        # Attention for bounding boxes
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.bbox_fc = nn.Linear(4, 512)  # Map bounding boxes to 512-dimensional features

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Classification layer
        self.fc = nn.Linear(512, num_activities)

    def forward(self, x, bounding_boxes):
        batch_size, seq_len, channels, height, width = x.size()

        # Process images through ResNet
        x = x.view(batch_size * seq_len, channels, height, width)
        features = self.resnet(x)
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
        
        features = self.dropout(features)  # Apply dropout to features
        
        # Process bounding boxes
        bbox_features = self.bbox_fc(bounding_boxes)

        # Combine image and bounding box features using attention
        combined_features, _ = self.attention(features, bbox_features, bbox_features)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined_features)
        
        transformer_output = self.dropout(transformer_output)  # Apply dropout before classification
        
        # Classification
        output = self.fc(transformer_output[:, -1, :])  # Use last sequence output

        return output

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
train_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/train_sequences_new.csv"
val_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/val_sequences_new.csv"
test_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/test_sequences_new.csv"
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"

# Datasets and DataLoaders
train_dataset = VehicleActivityDataset(csv_file=train_csv, frames_dir=frames_dir, transform=transform)
val_dataset = VehicleActivityDataset(csv_file=val_csv, frames_dir=frames_dir, transform=transform)
test_dataset = VehicleActivityDataset(csv_file=test_csv, frames_dir=frames_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Adjusted class weights for imbalanced data
class_weights = torch.tensor([1.0, 3.0, 1.5, 2.0, 2.5]).to(device)  # Example: Assign higher weight to underrepresented classes
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Model, Loss, Optimizer
model = VehicleActivityModel(num_activities=len(activity_mapping), seq_len=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Function to calculate Productivity of Equipment (POE)
def calculate_poe(activities):
    total_productivity = sum(activity_productivity[activity.item()] for activity in activities)
    operating_hours = len(activities)
    return total_productivity / operating_hours if operating_hours > 0 else 0

# Training Loop
for epoch in range(25):
    model.train()
    train_loss = 0
    for images, target, bounding_boxes in train_loader:
        images, target, bounding_boxes = images.to(device), target.to(device), bounding_boxes.to(device)
        optimizer.zero_grad()
        outputs = model(images, bounding_boxes)
        loss = criterion(outputs, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    # Step the scheduler
    scheduler.step()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, target, bounding_boxes in val_loader:
            images, target, bounding_boxes = images.to(device), target.to(device), bounding_boxes.to(device)
            outputs = model(images, bounding_boxes)
            loss = criterion(outputs, target)
            val_loss += loss.item()

        val_loss /= len(val_loader)

    # Testing Loop (after each epoch)
    model.eval()
    correct = 0
    total = 0
    poe_actual_values = []  # Actual POE values
    poe_predicted_values = []  # Predicted POE values
    with torch.no_grad():
        for images, target, bounding_boxes in test_loader:
            images, target, bounding_boxes = images.to(device), target.to(device), bounding_boxes.to(device)
            outputs = model(images, bounding_boxes)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Calculate actual and predicted productivity
            actual_poe = calculate_poe(target)
            predicted_poe = calculate_poe(predicted)
            poe_actual_values.append(actual_poe)
            poe_predicted_values.append(predicted_poe)

            # Debugging outputs and predictions
            print(f"Predicted Activities: {predicted.cpu().numpy()}")
            print(f"Actual Activities: {target.cpu().numpy()}")
            print(f"Actual POE: {actual_poe:.2f}, Predicted POE: {predicted_poe:.2f}")

    test_accuracy = 100 * correct / total
    avg_actual_poe = sum(poe_actual_values) / len(poe_actual_values) if poe_actual_values else 0
    avg_predicted_poe = sum(poe_predicted_values) / len(poe_predicted_values) if poe_predicted_values else 0

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Avg Actual POE: {avg_actual_poe:.2f}, Avg Predicted POE: {avg_predicted_poe:.2f}')

# Save the model
torch.save(model.state_dict(), "vehicle_activity_model.pth")
print("Model saved as 'vehicle_activity_model.pth'")

