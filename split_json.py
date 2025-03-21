import json
from sklearn.model_selection import train_test_split

# Paths
label_mapping_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/label_mapping.json"  # Input label mapping
train_output_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/subset_train_label_mapping.json"  # Output training mapping
#val_output_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/val_label_mapping.json"  # Output validation mapping
#test_output_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Python Script/Sequential_learning/test_label_mapping.json"  # Output testing mapping

# Load the label mapping
with open(label_mapping_path, "r") as infile:
    label_mapping = json.load(infile)

# Get all video sequence keys
all_sequences = list(label_mapping.keys())

# Ensure reproducibility by sorting (optional)
all_sequences = sorted(all_sequences)

# Split into train (80%) and temp (20%)
#train_sequences, temp_sequences = train_test_split(all_sequences, test_size=0.2, random_state=42)
train_sequences = [all_sequences[0]] 
# Split temp into validation (10%) and test (10%)
#val_sequences, test_sequences = train_test_split(temp_sequences, test_size=0.5, random_state=42)

# Create split mappings

train_label_mapping = {seq: label_mapping[seq] for seq in train_sequences}
#val_label_mapping = {seq: label_mapping[seq] for seq in val_sequences}
#test_label_mapping = {seq: label_mapping[seq] for seq in test_sequences}

# Save the split mappings to JSON files
with open(train_output_path, "w") as train_file:
    json.dump(train_label_mapping, train_file, indent=4)

#with open(val_output_path, "w") as val_file:
#    json.dump(val_label_mapping, val_file, indent=4)

#with open(test_output_path, "w") as test_file:
#    json.dump(test_label_mapping, test_file, indent=4)

# Print confirmation and details
print("Dataset successfully split!")
print(f"Training Sequences ({len(train_sequences)}): {train_sequences}")
#print(f"Validation Sequences ({len(val_sequences)}): {val_sequences}")
#print(f"Test Sequences ({len(test_sequences)}): {test_sequences}")

