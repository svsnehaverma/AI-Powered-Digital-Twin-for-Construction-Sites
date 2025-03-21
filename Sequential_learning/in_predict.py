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

def evaluate_test_model_per_sequence(
    model, test_loader, device, num_equipment=4, num_classes=5, save_dir="./predictions/"
):
    print("\nEvaluating the model on the test dataset...")
    model.eval()

    all_actual_productivity = []
    all_predicted_productivity = []
    all_per_sequence_results = []  # Store per-sequence results for analysis

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for batch_idx, (images, activity_labels, productivity_labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
            images = images.to(device)
            activity_labels = activity_labels.to(device)
            productivity_labels = productivity_labels.to(device)

            # Forward pass
            activity_preds, productivity_preds = model(images)

            batch_results = []

            # Process sequences in the batch
            for seq_idx in range(images.size(0)):  # Iterate through sequences in the batch
                seq_actual_productivity = productivity_labels[seq_idx].mean().item()
                seq_predicted_productivity = productivity_preds[seq_idx].item()

                seq_results = {"sequence_idx": batch_idx * test_loader.batch_size + seq_idx + 1}
                seq_results["actual_productivity"] = seq_actual_productivity
                seq_results["predicted_productivity"] = seq_predicted_productivity
                seq_results["activity_accuracy"] = {}

                # Process activities for each equipment
                for eq in range(num_equipment):
                    preds = activity_preds[seq_idx, :, eq, :]  # Shape: (Sequence Length, num_classes)
                    preds = torch.argmax(preds, dim=-1)  # Predicted classes for the sequence
                    targets = activity_labels[seq_idx, :, eq]  # Actual classes

                    # Calculate accuracy for valid frames (ignore -1)
                    valid_mask = targets != -1
                    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
                    total = valid_mask.sum().item()

                    if total > 0:
                        accuracy = correct / total
                    else:
                        accuracy = float("nan")  # No valid labels for this equipment

                    seq_results["activity_accuracy"][f"equipment_{eq + 1}"] = accuracy

                # Append to results
                batch_results.append(seq_results)

                # Store productivity for global tracking
                all_actual_productivity.append(seq_actual_productivity)
                all_predicted_productivity.append(seq_predicted_productivity)

            # Append batch results to global results
            all_per_sequence_results.extend(batch_results)

    # Save results
    results_path = os.path.join(save_dir, "per_sequence_results.json")
    with open(results_path, "w") as f:
        json.dump(all_per_sequence_results, f, indent=4)
    print(f"Per-sequence results saved to {results_path}")

    # Print summary
    print("\nPer-Sequence Results (First 5 Sequences):")
    for seq_result in all_per_sequence_results[:5]:
        print(
            f"Sequence {seq_result['sequence_idx']} - "
            f"Actual Productivity: {seq_result['actual_productivity']:.4f}, "
            f"Predicted Productivity: {seq_result['predicted_productivity']:.4f}"
        )
        for eq, acc in seq_result["activity_accuracy"].items():
            print(f"  {eq}: Activity Accuracy = {acc * 100:.2f}%")

    return all_per_sequence_results, all_actual_productivity, all_predicted_productivity


# Main Testing Script
if __name__ == "__main__":
    # Load the saved model
    model_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/trained model/model_epoch_2.pth"
    model = CNNTransformerModel(num_classes=5, num_equipment=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    
    # Dataset and DataLoader
    frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce"
    test_label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/test_label_mapping.json"

    with open(test_label_mapping_path, "r") as test_file:
        test_label_mapping = json.load(test_file)

    transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_dataset = ActivityDataset(frames_dir, test_label_mapping, transform=transform, sequence_length=10)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate the model and save predictions
    save_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/predictions/"
    (
    per_sequence_results,
    all_actual_productivity,
    all_predicted_productivity,
    ) = evaluate_test_model_per_sequence(model, test_loader, device, save_dir=save_dir)


    print(f"Predictions saved to {save_dir}")

