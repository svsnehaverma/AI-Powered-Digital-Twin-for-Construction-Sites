import os
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Define Dataset Class
class ActivityDataset(Dataset):
    def __init__(self, frames_dir, label_mapping, transform=None, sequence_length=10):
        self.frames_dir = frames_dir
        self.label_mapping = label_mapping
        self.transform = transform
        self.sequence_length = sequence_length
        self.data = self._prepare_data()

        # Activity-to-integer mapping
        self.activity_mapping = {
            "Idle": 0,
            "Swing Bucket": 1,
            "Load Bucket": 2,
            "Dump": 3,
            "Move": 4,
            "Unknown": -1  # Default value for missing activities
        }

        # Fixed equipment labels
        self.equipment_labels = ["excavator", "truck1", "truck2", "truck3"]

    def _prepare_data(self):
        data = []
        for folder_name, frames in self.label_mapping.items():
            sorted_frames = sorted(frames.keys())
            for i in range(len(sorted_frames) - self.sequence_length + 1):
                sequence = sorted_frames[i:i + self.sequence_length]
                data.append((folder_name, sequence))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder_name, sequence = self.data[idx]
        images, activity_labels, productivity_labels = [], [], []

        for frame_number in sequence:
            frame_info = self.label_mapping[folder_name][frame_number]
            frame_path = os.path.join(self.frames_dir, folder_name, f"{folder_name}_I{int(frame_number):05d}.jpg")
            
            # Load and transform the image
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            images.append(image)

            # Map activities to integers
            activity_dict = frame_info['activities']
            activity_row = [
                self.activity_mapping.get(activity_dict.get(equipment, "Unknown"), -1)
                for equipment in self.equipment_labels
            ]
            activity_labels.append(activity_row)

            # Collect productivity labels
            productivity_labels.append(frame_info['productivity'])

        # Convert to tensors
        images = torch.stack(images)  # Shape: (sequence_length, Channels, Height, Width)
        activity_labels = torch.tensor(activity_labels, dtype=torch.long)  # Shape: (sequence_length, num_equipment)
        productivity_labels = torch.tensor(productivity_labels, dtype=torch.float)  # Shape: (sequence_length,)

        return images, activity_labels, productivity_labels


# Define Model Architecture
class CNNTransformerModel(nn.Module):
    def __init__(self, cnn_out_dim=512, transformer_dim=512, n_heads=8, num_layers=4, num_classes=5, num_equipment=4):
        super(CNNTransformerModel, self).__init__()
        # CNN Backbone
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, cnn_out_dim)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=transformer_dim, nhead=n_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layers
        self.fc_activity = nn.ModuleList([nn.Linear(transformer_dim, num_classes) for _ in range(num_equipment)])
        self.fc_productivity = nn.Linear(transformer_dim, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # (B, T, C, H, W)
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten batch and sequence for CNN
        cnn_features = self.cnn(x)  # (B*T, cnn_out_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # Reshape back to (B, T, cnn_out_dim)

        transformer_features = self.transformer(cnn_features)  # (B, T, transformer_dim)

        # Activity predictions
        activity_preds = torch.stack(
            [fc(transformer_features) for fc in self.fc_activity], dim=2
        )  # Shape: (B, T, num_equipment, num_classes)

        # Productivity prediction
        productivity_preds = self.fc_productivity(transformer_features.mean(dim=1))  # (B, 1)

        return activity_preds, productivity_preds


# Training Pipeline
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, num_equipment=4):
    print("Initializing training...")
    # Losses
    criterion_activity = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_productivity = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss, total_activity_loss, total_productivity_loss = 0, 0, 0  # Initialize cumulative losses
        for batch_idx, (images, activity_labels, productivity_labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)} in training...")
           
            # Debugging inputs in the training loop
            print(f"Images Min: {images.min()}, Max: {images.max()}")
            print(f"Activity Labels Min: {activity_labels.min()}, Max: {activity_labels.max()}")
	    #print(f"Productivity Labels Min: {productivity_labels.min()}, Max: {productivity_labels.max()}")

            images = images.to(device)
            activity_labels = activity_labels.to(device)
            productivity_labels = productivity_labels.to(device)

            # Forward pass
            #print("Performing forward pass...")
            activity_preds, productivity_preds = model(images)
            
            # Clamp predictions to prevent instability
            activity_preds = torch.clamp(activity_preds, min=-10, max=10)
            
            # Debugging predictions
            print(f"Activity Predictions Min: {activity_preds.min()}, Max: {activity_preds.max()}")
            print(f"Productivity Predictions Min: {productivity_preds.min()}, Max: {productivity_preds.max()}")
            print(f"Unique Activity Labels: {torch.unique(activity_labels)}")

            # Compute activity loss for each equipment
            loss_activity = 0
            for i in range(num_equipment):
                preds = activity_preds[:, :, i, :]  # Shape: (B*T, num_classes)
                targets = activity_labels[:, :, i].reshape(-1)  # Shape: (B*T)
                 
                 # Skip equipment with all invalid labels (-1)
                if (targets != -1).sum() == 0:
                    print(f"Skipping Equipment {i} with all invalid labels.")
                    continue
        
                # Check for invalid labels
                print(f"Unique Activity Labels for Equipment {i}: {targets.unique()}")

                # Compute the loss for valid labels
                loss_activity += criterion_activity(preds.reshape(-1, preds.size(-1)), targets)

            # Compute productivity loss
            loss_productivity = criterion_productivity(productivity_preds.squeeze(), productivity_labels.mean(dim=1))
            
            # Combine losses
            loss = loss_activity + loss_productivity

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_activity_loss += loss_activity.item()
            total_productivity_loss += loss_productivity.item()

            # Print batch-wise losses
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, "
                f"Activity Loss: {loss_activity.item():.4f}, Productivity Loss: {loss_productivity.item():.4f}"
            )

        # Print cumulative losses for the epoch
        print(
            f"Epoch {epoch + 1}/{epochs} - Training Loss: {total_loss / len(train_loader):.4f}, "
            f"Activity Loss: {total_activity_loss / len(train_loader):.4f}, "
            f"Productivity Loss: {total_productivity_loss / len(train_loader):.4f}"
        )

        
        # Save the model after each epoch
        save_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/"  # You can change this to your desired directory
        save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
       
        # Validation
        print("Starting validation...")
        model.eval()
        with torch.no_grad():
            val_loss, val_activity_loss, val_productivity_loss = 0, 0, 0  # Initialize validation losses
            for batch_idx, (images, activity_labels, productivity_labels) in enumerate(val_loader):
                print(f"Processing batch {batch_idx + 1}/{len(val_loader)} in validation...")
                images = images.to(device)
                activity_labels = activity_labels.to(device)
                productivity_labels = productivity_labels.to(device)

                
                activity_preds, productivity_preds = model(images)

                # Compute activity loss for each equipment
                loss_activity = 0
                for i in range(num_equipment):
                    preds = activity_preds[:, :, i, :]
                    targets = activity_labels[:, :, i].reshape(-1)
                    loss_activity += criterion_activity(preds.reshape(-1, preds.size(-1)), targets)

                # Compute productivity loss
                loss_productivity = criterion_productivity(productivity_preds.squeeze(), productivity_labels.mean(dim=1))

                # Accumulate validation losses
                val_loss += loss.item()
                val_activity_loss += loss_activity.item()
                val_productivity_loss += loss_productivity.item()
                
                # Print actual and predicted productivity for each sequence
                print("Actual vs Predicted Productivity:")
                for i in range(images.size(0)):  # Iterate through the batch
                    actual = productivity_labels[i].mean().item()  # Average productivity for the sequence
                    predicted = productivity_preds[i].item()  # Predicted productivity for the sequence
                    print(f"Sequence {batch_idx * val_loader.batch_size + i + 1}: Actual = {actual:.4f}, Predicted = {predicted:.4f}")

                # Print batch-wise validation losses
                print(
                    f"Batch {batch_idx + 1}/{len(val_loader)} - Validation Loss: {loss_activity + loss_productivity:.4f}, "
                    f"Activity Loss: {loss_activity.item():.4f}, Productivity Loss: {loss_productivity.item():.4f}"
                )
                

            # Print validation losses for the epoch
            print(
                f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss / len(val_loader):.4f}, "
                f"Activity Loss: {val_activity_loss / len(val_loader):.4f}, "
                f"Productivity Loss: {val_productivity_loss / len(val_loader):.4f}"
            )

    # Save the final model
    final_save_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

    print("Training completed successfully!")            
            

# Main Script
if __name__ == "__main__":
    # Directories and Parameters
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
    train_label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/train_label_mapping.json"
    val_label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/val_label_mapping.json"
    test_label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/test_label_mapping.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    sequence_length = 10
    epochs = 25
    
    # Load Label Mappings
    with open(train_label_mapping_path, "r") as train_file:
        train_label_mapping = json.load(train_file)

    with open(val_label_mapping_path, "r") as val_file:
        val_label_mapping = json.load(val_file)

    with open(test_label_mapping_path, "r") as test_file:
        test_label_mapping = json.load(test_file)
        
    '''
    # Split Dataset
    from sklearn.model_selection import train_test_split



    # Ensure an 8-1-1 split
    all_sequences = list(label_mapping.keys())
    num_sequences = len(all_sequences)  # Total number of sequences
    #train_size = int(0.8 * num_sequences)  # 80% for training
    #val_size = 1  # 10% for validation (1 sequence)
    #test_size = 1  # 10% for testing (1 sequence)

    # Shuffle the sequences
    all_sequences = sorted(all_sequences)  # Ensure consistent order
    train_sequences = all_sequences[:train_size]
    val_sequences = all_sequences[train_size:train_size + val_size]
    test_sequences = all_sequences[train_size + val_size:]

    # Create label mappings for each set
    train_label_mapping = {seq: label_mapping[seq] for seq in train_sequences}
    val_label_mapping = {seq: label_mapping[seq] for seq in val_sequences}
    test_label_mapping = {seq: label_mapping[seq] for seq in test_sequences}


    # Prepare data indices (all sequences are folder names in label_mapping)
    all_sequences = list(label_mapping.keys())  # All folder names (sequences)

    # Split sequences into train, validation, and test sets
    train_sequences, test_sequences = train_test_split(all_sequences, test_size=0.2, random_state=42)
    train_sequences, val_sequences = train_test_split(train_sequences, test_size=0.5, random_state=42)

    # Create label mappings for each set
    train_label_mapping = {seq: label_mapping[seq] for seq in train_sequences}
    val_label_mapping = {seq: label_mapping[seq] for seq in val_sequences}
    test_label_mapping = {seq: label_mapping[seq] for seq in test_sequences}
    '''
    # Print information about the dataset splits
    print("\nDataset Splitting:")
    print(f"Training Sequences ({len(train_label_mapping)}): {list(train_label_mapping.keys())}")
    print(f"Validation Sequences ({len(val_label_mapping)}): {list(val_label_mapping.keys())}")
    print(f"Test Sequences ({len(test_label_mapping)}): {list(test_label_mapping.keys())}")


    # Dataset and DataLoader
    transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    train_dataset = ActivityDataset(frames_dir, train_label_mapping, transform, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = ActivityDataset(frames_dir, val_label_mapping, transform, sequence_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ActivityDataset(frames_dir, test_label_mapping, transform, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model Initialization
    model = CNNTransformerModel(num_classes=5, num_equipment=4).to(device)

    # Train the Model
    train_model(model, train_loader, val_loader, device, epochs=epochs, lr=1e-5)

    # Test the Model
    def test_model(model, test_loader, device):
        print("\nTesting the model...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, activity_labels, productivity_labels) in enumerate(test_loader):
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)} in testing...")
                images = images.to(device)
                activity_labels = activity_labels.to(device)
                productivity_labels = productivity_labels.to(device)
                

                # Forward pass
                activity_preds, productivity_preds = model(images)

                # Print Actual and Predicted Productivity for Testing
                print("Test Results: Actual vs Predicted Productivity")
                for i in range(images.size(0)):
                    actual = productivity_labels[i].mean().item()
                    predicted = productivity_preds[i].item()
                    print(f"Sequence {batch_idx * test_loader.batch_size + i + 1}: Actual = {actual:.4f}, Predicted = {predicted:.4f}")

    test_model(model, test_loader, device)

