import os
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Directories
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2/"
labels_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/Labels/"
annotations_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/DetectionTrackingAnnotations/"
output_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/"

# Sliding window parameters
seq_len = 25  # Number of frames per sequence
step = 10     # Step size for sliding window

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

# Function to extract bounding boxes from XML
def get_bounding_boxes(xml_file, frame_number):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bounding_boxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        start_frame = int(obj.find('startFrame').text)
        end_frame = int(obj.find('endFrame').text)

        if start_frame <= frame_number <= end_frame:
            polygon = obj.find('polygon')
            if polygon is not None:
                x_values = [int(pt.find('x').text) for pt in polygon.findall('pt')]
                y_values = [int(pt.find('y').text) for pt in polygon.findall('pt')]

                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
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
        valid_labels = set()
        
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            equipment = os.path.splitext(label_file)[0]
            activities[equipment] = load_activities(label_path)
            valid_labels.add(equipment)
        
        print(f"Valid labels found: {valid_labels}")

        # Find the corresponding XML file in the annotations folder
        xml_file = os.path.join(annotations_dir, f"{folder_name}.xml")
        if not os.path.exists(xml_file):
            print(f"No XML file found for {folder_name}")
            continue
        
        # Process all frames in the folder
        for frame_file in os.listdir(folder_path):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(folder_path, frame_file)
                print(f"Processing file: {frame_file}")
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

# Convert the collected data into a DataFrame
df = pd.DataFrame(data)
print(f"Processed {len(df)} frames.")

# Function to create sliding window sequences
def create_sequences(data, seq_len, step=1):
    sequences = []
    for i in range(0, len(data) - seq_len + 1, step):
        seq_frames = data[i:i+seq_len]

        combined_activities = set()
        combined_equipments = set()
        combined_bboxes = []
        
        for frame in seq_frames:
            combined_activities.update(frame["activities"])
            combined_equipments.update(frame["equipment"])
            combined_bboxes.extend(frame["bounding_boxes"])

        sequences.append({
            "frames": [frame["frame"] for frame in seq_frames],
            "equipment": list(combined_equipments),
            "activities": list(combined_activities),
            "bounding_boxes": combined_bboxes
        })
    return sequences

# Group data by video and create sequences
sequences = []
for folder_name, group in df.groupby(df['frame'].str.split('_').str[0]):  # Group by video
    video_data = group.to_dict('records')
    sequences.extend(create_sequences(video_data, seq_len, step))

# Convert sequences to DataFrame
seq_df = pd.DataFrame(sequences)

# Rearrange columns to place 'equipment' after 'frames'
seq_df = seq_df[['frames', 'equipment', 'activities', 'bounding_boxes']]

# Convert activities to multi-label format  to convert in binary format
#seq_df['multi_label'] = seq_df['activities'].apply(
#    lambda x: [1 if action in x else 0 for action in ['Idle', 'Swing Bucket', 'Load Bucket', 'Dump', 'Move']]
#)

# Split into train, validation, and test sets
train_size = 0.8
val_size = 0.1

train_idx = int(len(seq_df) * train_size)
val_idx = train_idx + int(len(seq_df) * val_size)

train_df = seq_df.iloc[:train_idx]
val_df = seq_df.iloc[train_idx:val_idx]
test_df = seq_df.iloc[val_idx:]

# Save the splits to CSV files
train_df.to_csv(os.path.join(output_dir, "train_sequences_new.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val_sequences_new.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_sequences_new.csv"), index=False)

print("Data splits saved successfully.")

