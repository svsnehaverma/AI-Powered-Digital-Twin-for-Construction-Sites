import os
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import numpy as np
from in_train import CNNTransformerModel
from in_train import ActivityDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test Function
# Test Function
def evaluate_test_model(model, test_loader, device, num_equipment=4, num_classes=5, save_dir="./predictions/"):
    print("\nEvaluating the model on the test dataset...")
    model.eval()

    all_actual_activities = []
    all_predicted_activities = []
    all_actual_productivity = []
    all_predicted_productivity = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    activity_accuracy = {i: [] for i in range(num_equipment)}  # Accuracy per equipment
    productivity_loss = []
    
    all_per_sequence_results_epoch1 = []  # Initialize the list to store per-sequence results

    with torch.no_grad():
        for batch_idx, (images, activity_labels, productivity_labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
            images = images.to(device)
            activity_labels = activity_labels.to(device)
            productivity_labels = productivity_labels.to(device)

            # Forward pass
            activity_preds, productivity_preds = model(images)

            # Process activities
            for eq in range(num_equipment):
                preds = activity_preds[:, :, eq, :]  # (Batch Size, Sequence Length, num_classes)
                preds = torch.argmax(preds, dim=-1)  # Predicted classes for each frame
                targets = activity_labels[:, :, eq]  # Actual classes

                # Append to global lists
                all_actual_activities.append(targets.cpu().numpy())
                all_predicted_activities.append(preds.cpu().numpy())

                # Calculate accuracy for valid frames (ignore -1)
                valid_mask = targets != -1
                correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                if total > 0:
                    accuracy = correct / total
                    activity_accuracy[eq].append(accuracy)

            # Process productivity
            actual_productivity = productivity_labels.mean(dim=1).cpu().numpy()  # Sequence-level actual productivity
            predicted_productivity = productivity_preds.squeeze().cpu().numpy()  # Sequence-level predicted productivity
            loss = np.abs(actual_productivity - predicted_productivity)  # Absolute difference in productivity

            # Append to global lists
            all_actual_productivity.extend(actual_productivity)
            all_predicted_productivity.extend(predicted_productivity)
            productivity_loss.extend(loss)

            # Collect per-sequence results for the current batch
            for seq_idx in range(images.size(0)):  # Loop over the batch size (sequences)
                per_sequence_result = {
                    "sequence_idx": seq_idx,
                    "actual_productivity": actual_productivity[seq_idx],
                    "predicted_productivity": predicted_productivity[seq_idx],
                    "productivity_loss": loss[seq_idx],
                    "activity_accuracy": {eq: activity_accuracy[eq][seq_idx] for eq in range(num_equipment)},
                }
                all_per_sequence_results_epoch1.append(per_sequence_result)

    # Calculate overall accuracy for activities
    avg_activity_accuracy = {eq: np.mean(accuracy_list) for eq, accuracy_list in activity_accuracy.items()}
    
    # Save results
    results_path = os.path.join(save_dir, "per_sequence_results_epoch1.json")
    with open(results_path, "w") as f:
        json.dump(all_per_sequence_results_epoch1, f, indent=4)
    print(f"Per-sequence results saved to {results_path}")

    print("\nTest Results:")
    print("Activity Accuracy per Equipment:")
    for eq, acc in avg_activity_accuracy.items():
        print(f"Equipment {eq + 1}: {acc * 100:.2f}%")
    print(f"Average Productivity Loss: {np.mean(productivity_loss):.4f}")

    return (
        avg_activity_accuracy,
        all_actual_activities,
        all_predicted_activities,
        all_actual_productivity,
        all_predicted_productivity,
        productivity_loss,
    )



# Visualization Function
def plot_test_results(
    all_actual_activities, all_predicted_activities, all_actual_productivity, all_predicted_productivity, productivity_loss):
    num_sequences = len(all_actual_productivity)

    # Plot productivity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_sequences), all_actual_productivity, label="Actual Productivity", marker="o")
    plt.plot(range(num_sequences), all_predicted_productivity, label="Predicted Productivity", marker="x")
    plt.xlabel("Sequence Index")
    plt.ylabel("Productivity")
    plt.title("Actual vs Predicted Productivity")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot productivity loss
    plt.figure(figsize=(10, 6))
    plt.bar(sequence_indices, predicted_productivity, color="orange", alpha=0.7, label="Predicted Productivity")
    plt.xlabel("Sequence Index",  fontsize=14, fontweight="bold")
    plt.ylabel("Absolute Predicted loss",  fontsize=14, fontweight="bold")
    plt.title("Predicted Productivity loss per Sequence",  fontsize=14, fontweight="bold")
    # Set bold font for axis tick labels
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    plt.legend(fontsize=12, frameon=False, loc="best", prop={"weight": "bold"})
    plt.tight_layout()
    plt.show()

    # Plot actual vs predicted activity sequences for the first batch
    if len(all_actual_activities) > 0:
        actual_seq = np.concatenate(all_actual_activities[:1], axis=1)  # First batch, all equipment
        predicted_seq = np.concatenate(all_predicted_activities[:1], axis=1)  # First batch, all equipment
        equipment_labels = ["Excavator", "Truck1", "Truck2", "Truck3"]

        plt.figure(figsize=(12, 8))
        for eq in range(len(equipment_labels)):
            plt.subplot(len(equipment_labels), 1, eq + 1)
            plt.plot(actual_seq[eq], label="Actual", marker="o", linestyle="--")
            plt.plot(predicted_seq[eq], label="Predicted", marker="x", linestyle="-")
            plt.title(f"Activity Sequence for {equipment_labels[eq]}")
            plt.xlabel("Frame Index")
            plt.ylabel("Activity Class")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()


# Main Testing Script
if __name__ == "__main__":
    # Load the saved model
    model_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/trained model/model_epoch_1.pth"
    model = CNNTransformerModel(num_classes=5, num_equipment=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # Dataset and DataLoader
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
    test_label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/test_label_mapping.json"

    with open(test_label_mapping_path, "r") as test_file:
        test_label_mapping = json.load(test_file)

    transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_dataset = ActivityDataset(frames_dir, test_label_mapping, transform=transform, sequence_length=10)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate the model
    # Evaluate the model and save predictions
    save_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/predictions/"
    (
        avg_activity_accuracy,
        all_actual_activities,
        all_predicted_activities,
        all_actual_productivity,
        all_predicted_productivity,
        productivity_loss,
    ) = evaluate_test_model(model, test_loader, device)

    # Plot the results
    plot_test_results(
        all_actual_activities,
        all_predicted_activities,
        all_actual_productivity,
        all_predicted_productivity,
        productivity_loss,
    )
