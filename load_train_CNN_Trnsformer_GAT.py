import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Activity mapping
activity_mapping = {
    'Idle': 0,
    'Swing Bucket': 1,
    'Load Bucket': 2,
    'Dump': 3,
    'Move': 4
}

# Dataset class
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
        return matching_files[0] if matching_files else None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frame_files = eval(row["frames"])
        images = [self.transform(cv2.imread(self.find_image_path(frame))) for frame in frame_files]
        images = torch.stack(images)
        activities = eval(row["activities"])
        binary_activities = torch.zeros(len(activity_mapping), dtype=torch.float32)
        for activity in activities:
            if activity in activity_mapping:
                binary_activities[activity_mapping[activity]] = 1.0
        return images, binary_activities

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model definition
class ActivityTransformer(nn.Module):
    def __init__(self, num_classes, seq_len=10):
        super(ActivityTransformer, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = self.pool(features).squeeze()
        features = features.view(batch_size, seq_len, -1).permute(1, 0, 2)
        transformed_features = self.transformer(features)
        output = self.fc(transformed_features[-1])
        return output

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "activity_transformer_model.pth")
    print("Model saved to activity_transformer_model.pth")

if __name__ == "__main__":
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
    train_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/train_subset.csv"
    val_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/val_subset.csv"
    train_dataset = EquipmentActivityDataset(train_csv, frames_dir, transform)
    val_dataset = EquipmentActivityDataset(val_csv, frames_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    model = ActivityTransformer(num_classes=len(activity_mapping))
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, train_loader, val_loader, criterion, optimizer)

