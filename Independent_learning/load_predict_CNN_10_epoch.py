import torch
import pandas as pd
import numpy as np
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

# Activity mapping dictionary
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

        # Convert activity strings to binary multi-label format
        binary_activities = torch.zeros(len(activity_mapping), dtype=torch.float32)
        for activity_str in activities:
            if activity_str in activity_mapping:
                activity_int = activity_mapping[activity_str]
                binary_activities[activity_int] = 1.0

        return image, binary_activities

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model definition
class ActivityCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ActivityCNN, self).__init__()
        self.extra_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.extra_conv(x)
        x = self.model(x)
        return x

# Load the saved model
def load_model(model_path, num_classes=5):
    model = ActivityCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

# Function to make predictions
def make_predictions(model, test_loader):
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.numpy())

    predictions = np.vstack(predictions)
    ground_truth = np.vstack(ground_truth)
    return predictions, ground_truth

# Map predictions to activity labels
def map_predictions_to_labels(predictions, activity_mapping):
    reverse_mapping = {v: k for k, v in activity_mapping.items()}
    activity_labels = []
    for pred in predictions:
        labels = [reverse_mapping[i] for i, val in enumerate(pred) if val == 1.0]
        activity_labels.append(labels)
    return activity_labels



def visualize_predictions(test_dataset, predictions, ground_truth, activity_mapping, num_samples=5):
    reverse_mapping = {v: k for k, v in activity_mapping.items()}
    indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)  # Randomly pick samples

    for idx in indices:
        # Get image, ground truth, and prediction
        image, true_labels = test_dataset[idx]
        predicted_labels = predictions[idx]
        frame_file = test_dataset.data.iloc[idx]["frame"]  # Get the image filename

        # Convert tensor to numpy and de-normalize for visualization
        img = image.numpy().transpose(1, 2, 0)  # Convert CHW to HWC
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # De-normalize
        img = np.clip(img, 0, 1)  # Clip values for valid image range

        # Map ground truth and predictions to labels
        true_activities = [reverse_mapping[i] for i, val in enumerate(true_labels) if val == 1.0]
        pred_activities = [reverse_mapping[i] for i, val in enumerate(predicted_labels) if val == 1.0]

        # Plot the image with labels
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Filename: {frame_file}\nGround Truth: {true_activities}\nPredicted: {pred_activities}", fontsize=10)
        plt.show()

if __name__ == "__main__":
    # Paths
    test_csv = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Independent_Splited_data/test_dataset.csv"
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
    model_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/activity_cnn_model_multilabel_10.pth"

    # Load the dataset and DataLoader
    test_dataset = EquipmentActivityDataset(csv_file=test_csv, frames_dir=frames_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Load the trained model
    model = load_model(model_path, num_classes=len(activity_mapping))

    # Make predictions
    predictions, ground_truth = make_predictions(model, test_loader)

    # Map binary predictions to activity labels
    predicted_labels = map_predictions_to_labels(predictions, activity_mapping)
    
    # Visualize predictions
    visualize_predictions(test_dataset, predictions, ground_truth, activity_mapping, num_samples=5)

    # Print results
    print("Sample Predictions (binary format):")
    print(predictions[:5])

    print("Sample Ground Truth:")
    print(ground_truth[:5])

    print("Sample Predicted Labels:")
    print(predicted_labels[:5])

