'''
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the activity mapping
activity_mapping = {
    'Idle': 0,
    'Swing Bucket': 1,
    'Load Bucket': 2,
    'Dump': 3,
    'Move': 4
}

# Define your dataset class
class EquipmentActivityDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, frames_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)  # Number of samples

    def __getitem__(self, idx):
        # Convert activities from multi-label binary format to binary vector
        multi_label = eval(self.data.iloc[idx]["multi_label"])  # Convert the string list to an actual list
        # Ensure we have the correct number of activity labels (5 binary labels)
        activity_vector = torch.tensor(multi_label, dtype=torch.float32)
        
        # If multi_label has missing or wrong size, we set it to a zero vector (fallback)
        if activity_vector.size(0) != len(activity_mapping):
            activity_vector = torch.zeros(len(activity_mapping), dtype=torch.float32)

        return activity_vector

# Load your dataset
csv_file = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/train_dataset.csv"
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
dataset = EquipmentActivityDataset(csv_file=csv_file, frames_dir=frames_dir)

# Prepare the list of activity vectors
activity_vectors = []

# Collect activity vectors
for idx in range(len(dataset)):
    activity_vector = dataset[idx]  # Get the activity vector (binary)
    # Skip empty activity vectors
    if activity_vector.sum() > 0:
        activity_vectors.append(activity_vector.numpy())  # Convert to numpy for easier handling

# Convert to a 2D numpy array (samples x activities)
activity_matrix = np.array(activity_vectors)

# Check the shape of activity_matrix for debugging
print(f"Shape of activity_matrix: {activity_matrix.shape}")

# Ensure activity_matrix is 2D (samples x activities)
if len(activity_matrix.shape) == 1:
    # Reshape if only a single dimension (one activity vector)
    activity_matrix = activity_matrix.reshape(-1, len(activity_mapping))

# Compute the correlation matrix
correlation_matrix = np.corrcoef(activity_matrix.T)  # Transpose to get activities as columns

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=list(activity_mapping.keys()), yticklabels=list(activity_mapping.keys()))
plt.title('Activity Correlation Matrix')
plt.show()
'''
