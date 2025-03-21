'''
################## Seprate the train test and validation sett randomaly ##############
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Directories
frames_dir = "/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/frames_ce_2/"
labels_dir = "/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/Labels/"
annotations_dir = "/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/DetectionTrackingAnnotations/"

# Function to load activities from file
def load_activities(file_path):
    activities = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.strip():  # Ensure the line is not empty
                parts = line.strip().split()
                if len(parts) == 3:
                    frame_start, frame_end, action_type = map(int, parts)
                    action_description = {0: 'Idle', 1: 'Swing Bucket', 2: 'Load Bucket', 3: 'Dump', 4: 'Move'}.get(action_type, 'Unknown')
                    activities.append((frame_start, frame_end, action_description))
    return activities

# Function to extract polygon and convert to bounding box from XML
def get_bounding_boxes(xml_file, frame_number):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bounding_boxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        start_frame = int(obj.find('startFrame').text)
        end_frame = int(obj.find('endFrame').text)

        # Ensure bounding box is valid for the current frame
        if start_frame <= frame_number <= end_frame:
            polygon = obj.find('polygon')
            if polygon is not None:
                x_values = []
                y_values = []
                for pt in polygon.findall('pt'):
                    x = int(pt.find('x').text)
                    y = int(pt.find('y').text)
                    x_values.append(x)
                    y_values.append(y)

                # Calculate the bounding box from the polygon points
                x_min = min(x_values)
                y_min = min(y_values)
                x_max = max(x_values)
                y_max = max(y_values)

                bounding_boxes.append({'label': label, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})
    return bounding_boxes

# Preprocess the data
data = []

# Iterate over each folder in the frames directory
for folder_name in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder_name)
    if os.path.isdir(folder_path):  # Ensure it’s a directory
        print(f"Processing folder: {folder_name}")
        
        # Check for corresponding label files in the Labels directory
        label_folder = os.path.join(labels_dir, folder_name)
        if not os.path.exists(label_folder):
            print(f"No label folder found for {folder_name}")
            continue
        
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
        activities = {}
        valid_labels = set()  # This will store the valid equipment labels (e.g., 'truck1', 'truck2', 'truck3')
        
        # Load activities from all label files in the current folder
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            equipment = os.path.splitext(label_file)[0]  # Get equipment name (excavator, truck1, etc.)
            print(f"Loading activities for {equipment}")
            activities[equipment] = load_activities(label_path)
            valid_labels.add(equipment)  # Add the equipment to the valid label set
        
        print(f"Valid labels found: {valid_labels}")  # Print the valid labels

        # Find the corresponding XML file in the annotations folder
        xml_file = os.path.join(annotations_dir, f"{folder_name}.xml")
        if not os.path.exists(xml_file):
            print(f"No XML file found for {folder_name}")
            continue
        
        # Process all frames in the folder
        for frame_file in os.listdir(folder_path):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(folder_path, frame_file)
                print(f"Processing frame: {frame_file}")
                
                try:
                    frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])  # Extract frame number
                except ValueError:
                    print(f"Unable to extract frame number from filename: {frame_file}")
                    continue

                # Match activity labels
                frame_activities = {}
                for equipment, activity_list in activities.items():
                    for (start, end, action) in activity_list:
                        if start <= frame_number <= end:
                            frame_activities[equipment] = action
                            print(f"Matched activity for equipment {equipment}: {action} (Start: {start}, End: {end})")
                            break

                # Get bounding boxes for the current frame
                bounding_boxes = get_bounding_boxes(xml_file, frame_number)

                # Store frame, activity, and bounding box information
                data.append({
                    "frame": frame_file,
                    "equipment": list(frame_activities.keys()),
                    "activities": list(frame_activities.values()),
                    "bounding_boxes": bounding_boxes
                })

# Convert the collected data into a DataFrame for easier manipulation
df = pd.DataFrame(data)
print(f"Processed {len(df)} frames.")

# Split the dataset into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# Display key information about each dataset split
print("Train Set:")
print(train_df.head())

print("Validation Set:")
print(val_df.head())

print("Test Set:")
print(test_df.head())

# Optional: Save the splits to CSV files for future use
train_df.to_csv("/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Splited_data/train_dataset.csv", index=False)
val_df.to_csv("/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Splited_data/val_dataset.csv", index=False)
test_df.to_csv("/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Splited_data/test_dataset.csv", index=False)
'''

# spilling with Multilabel statified Kfold or statifieshuffle split #######
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Directories
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce/"
labels_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/Labels/"
annotations_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/DetectionTrackingAnnotations/"

# Function to load activities from file
def load_activities(file_path):
    activities = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.strip():  # Ensure the line is not empty
                parts = line.strip().split()
                if len(parts) == 3:
                    frame_start, frame_end, action_type = map(int, parts)
                    action_description = {0: 'Idle', 1: 'Swing Bucket', 2: 'Load Bucket', 3: 'Dump', 4: 'Move'}.get(action_type, 'Unknown')
                    activities.append((frame_start, frame_end, action_description))
    return activities

# Function to extract polygon and convert to bounding box from XML
def get_bounding_boxes(xml_file, frame_number):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bounding_boxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        start_frame = int(obj.find('startFrame').text)
        end_frame = int(obj.find('endFrame').text)

        # Ensure bounding box is valid for the current frame
        if start_frame <= frame_number <= end_frame:
            polygon = obj.find('polygon')
            if polygon is not None:
                x_values = []
                y_values = []
                for pt in polygon.findall('pt'):
                    x = int(pt.find('x').text)
                    y = int(pt.find('y').text)
                    x_values.append(x)
                    y_values.append(y)

                # Calculate the bounding box from the polygon points
                x_min = min(x_values)
                y_min = min(y_values)
                x_max = max(x_values)
                y_max = max(y_values)

                bounding_boxes.append({'label': label, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})
    return bounding_boxes

# Preprocess the data
data = []

# Iterate over each folder in the frames directory
for folder_name in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder_name)
    if os.path.isdir(folder_path):  # Ensure it’s a directory
        print(f"Processing folder: {folder_name}")
        
        # Check for corresponding label files in the Labels directory
        label_folder = os.path.join(labels_dir, folder_name)
        if not os.path.exists(label_folder):
            print(f"No label folder found for {folder_name}")
            continue
        
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
        activities = {}
        valid_labels = set()  # This will store the valid equipment labels (e.g., 'truck1', 'truck2', 'truck3')
        
        # Load activities from all label files in the current folder
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            equipment = os.path.splitext(label_file)[0]  # Get equipment name (excavator, truck1, etc.)
            print(f"Loading activities for {equipment}")
            activities[equipment] = load_activities(label_path)
            valid_labels.add(equipment)  # Add the equipment to the valid label set
        
        print(f"Valid labels found: {valid_labels}")  # Print the valid labels

        # Find the corresponding XML file in the annotations folder
        xml_file = os.path.join(annotations_dir, f"{folder_name}.xml")
        if not os.path.exists(xml_file):
            print(f"No XML file found for {folder_name}")
            continue
        
        # Process all frames in the folder
        for frame_file in os.listdir(folder_path):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(folder_path, frame_file)
                print(f"Processing frame: {frame_file}")
                
                try:
                    frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])  # Extract frame number
                except ValueError:
                    print(f"Unable to extract frame number from filename: {frame_file}")
                    continue

                # Match activity labels
                frame_activities = {}
                for equipment, activity_list in activities.items():
                    for (start, end, action) in activity_list:
                        if start <= frame_number <= end:
                            frame_activities[equipment] = action
                            print(f"Matched activity for equipment {equipment}: {action} (Start: {start}, End: {end})")
                            break

                # Get bounding boxes for the current frame
                bounding_boxes = get_bounding_boxes(xml_file, frame_number)

                # Store frame, activity, and bounding box information
                data.append({
                    "frame": frame_file,
                    "equipment": list(frame_activities.keys()),
                    "activities": list(frame_activities.values()),
                    "bounding_boxes": bounding_boxes
                })

# Convert the collected data into a DataFrame for easier manipulation
df = pd.DataFrame(data)
print(f"Processed {len(df)} frames.")

# First, ensure each activity is converted to a binary multi-label vector
# Prepare the target labels as multi-label for stratified split
df['multi_label'] = df['activities'].apply(lambda x: [1 if action in x else 0 for action in ['Idle', 'Swing Bucket', 'Load Bucket', 'Dump', 'Move']])

# Convert 'multi_label' column to a numpy array for use in MultiLabelStratifiedShuffleSplit
multi_label_array = np.array(df['multi_label'].tolist())
print(multi_label_array)

# Perform stratified shuffle split for train/validation/test using MultiLabelStratifiedShuffleSplit
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(msss)

for train_idx, test_idx in msss.split(df, multi_label_array):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

# Split the test set into validation and test
msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for val_idx, test_idx in msss_val.split(test_df, np.array(test_df['multi_label'].tolist())):
    val_df = test_df.iloc[val_idx]
    test_df = test_df.iloc[test_idx]

# Display key information about each dataset split
print(f"Train set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Save the splits to CSV files for future use
train_df.to_csv("/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/small_train_dataset.csv", index=False)
val_df.to_csv("/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/small_val_dataset.csv", index=False)
test_df.to_csv("/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Splited_data/small_test_dataset.csv", index=False)
