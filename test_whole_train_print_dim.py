import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os

# Define a mapping from activity strings to integers
activity_mapping = {
    'Idle': 0,
    'Swing Bucket': 1,
    'Load Bucket': 2,
    'Dump': 3,
    'Move': 4
}

# Define transformations for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

        # Pad activity sequence if needed
        activity_seq = [self.activity_mapping[activity] for activity in activities]
        activity_seq += [0] * (self.max_seq_len - len(activity_seq))  # Pad with '0'
        activity_seq = activity_seq[:self.max_seq_len]
        target = activity_seq[-1]  # Take the last activity as the target

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

        return images, target, bbox_features

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
        
        # For ResNet18 convolutional layers, you can print details like:
        print("ResNet Conv1 kernel size and stride:")
        print(f"Kernel size: {self.resnet.conv1.kernel_size}")
        print(f"Stride: {self.resnet.conv1.stride}")

        # Optionally, you can loop through all the layers in ResNet and print their kernel sizes and strides
        for name, layer in self.resnet.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(f"{name} - Kernel size: {layer.kernel_size}, Stride: {layer.stride}")

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=6
        )

        # Scaled Dot-Product Attention for bounding boxes
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.bbox_fc = nn.Linear(4, 512)  # Map bounding boxes to 512-dimensional features

        # Classification layer
        self.fc = nn.Linear(512, num_activities)

    def forward(self, x, bounding_boxes):
        # Print input dimensions
        print(f"Input to model: {x.shape}")

        batch_size, seq_len, channels, height, width = x.size()

        # Process images through ResNet
        x = x.view(batch_size * seq_len, channels, height, width)
        print(f"Reshaped input to ResNet: {x.shape}")

        features = self.resnet(x)
        print(f"Features from ResNet: {features.shape}")

        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
        print(f"Reshaped features after ResNet: {features.shape}")

        # Process bounding boxes
        bbox_features = self.bbox_fc(bounding_boxes)  # Map bounding boxes to 512-dim
        print(f"BBox features: {bbox_features.shape}")

        # Combine image and bounding box features using attention
        combined_features, _ = self.attention(features, bbox_features, bbox_features)
        print(f"Combined features after attention: {combined_features.shape}")

        # Transformer Encoder
        transformer_output = self.transformer_encoder(combined_features)
        print(f"Output from transformer: {transformer_output.shape}")

        # Classification
        output = self.fc(transformer_output[:, -1, :])  # Use last sequence output
        print(f"Output from classification layer: {output.shape}")

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, Loss, Optimizer
model = VehicleActivityModel(num_activities=len(activity_mapping), seq_len=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    with torch.no_grad():
        for images, target, bounding_boxes in test_loader:
            images, target, bounding_boxes = images.to(device), target.to(device), bounding_boxes.to(device)
            outputs = model(images, bounding_boxes)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), "vehicle_activity_model.pth")
print("Model saved as 'vehicle_activity_model.pth'")



