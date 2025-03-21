import os
import xml.etree.ElementTree as ET

# Directories
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce"
labels_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/Labels"
annotations_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/DetectionTrackingAnnotations"

# Function to load activities from file
def load_activities(file_path):
    activities = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.strip():  # Ensure the line is not empty
                parts = line.strip().split()
                if len(parts) == 3:
                    frame_start, frame_end, action_type = map(int, parts)
                    action_description = {
                        0: 'Idle', 1: 'Swing Bucket', 2: 'Load Bucket',
                        3: 'Dump', 4: 'Move'
                    }.get(action_type, 'Unknown')
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

                bounding_boxes.append({
                    'label': label, 'x_min': x_min, 'y_min': y_min,
                    'x_max': x_max, 'y_max': y_max
                })
    return bounding_boxes

# Main function to create label_mapping
# Function to create label mapping with consistent equipment labels
def create_label_mapping(frames_dir, labels_dir, annotations_dir):
    label_mapping = {}

    # Fixed equipment labels
    equipment_labels = ["excavator", "truck1", "truck2", "truck3"]

    # Iterate over each folder in the frames directory
    for folder_name in os.listdir(frames_dir):
        folder_path = os.path.join(frames_dir, folder_name)
        if os.path.isdir(folder_path):
            label_mapping[folder_name] = {}

            # Load activities for the folder
            label_folder = os.path.join(labels_dir, folder_name)
            if not os.path.exists(label_folder):
                continue

            activities = {}
            label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
            for label_file in label_files:
                label_path = os.path.join(label_folder, label_file)
                equipment = os.path.splitext(label_file)[0]
                activities[equipment] = load_activities(label_path)

            # Load bounding boxes and match activities
            xml_file = os.path.join(annotations_dir, f"{folder_name}.xml")
            if os.path.exists(xml_file):
                for frame_file in os.listdir(folder_path):
                    if frame_file.endswith('.jpg'):
                        try:
                            frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])
                        except ValueError:
                            continue

                        # Match activities for the frame
                        frame_activities = {}
                        for equipment in equipment_labels:
                            if equipment in activities:
                                # Find the activity for this frame
                                for (start, end, action) in activities[equipment]:
                                    if start <= frame_number <= end:
                                        frame_activities[equipment] = action
                                        break
                            # Default to 'Unknown' if no activity found
                            if equipment not in frame_activities:
                                frame_activities[equipment] = "Unknown"

                        # Get bounding boxes
                        bounding_boxes = get_bounding_boxes(xml_file, frame_number)

                        # Ensure bounding boxes are mapped to equipment
                        filtered_bounding_boxes = []
                        truck_counter = 1  # Counter to assign truck labels
                        for bbox in bounding_boxes:
                            label = bbox["label"]
                            if label == "truck" and truck_counter <= 3:
                                bbox["label"] = f"truck{truck_counter}"
                                truck_counter += 1
                            filtered_bounding_boxes.append(bbox)

                        # Add missing bounding boxes for consistency
                        for equipment in equipment_labels:
                            if not any(bbox["label"] == equipment for bbox in filtered_bounding_boxes):
                                filtered_bounding_boxes.append(
                                    {"label": equipment, "x_min": None, "y_min": None, "x_max": None, "y_max": None}
                                )

                        # Placeholder for productivity label
                        productivity_label = 0  # Replace this if productivity data exists

                        # Save to label_mapping
                        label_mapping[folder_name][frame_number] = {
                            "activities": frame_activities,
                            "bounding_boxes": filtered_bounding_boxes,
                            "productivity": productivity_label,
                        }

    return label_mapping

# Generate the label mapping
label_mapping = create_label_mapping(frames_dir, labels_dir, annotations_dir)

print(label_mapping)
# Save or print the label_mapping for verification
import json
with open("testing_label_mapping.json", "w") as outfile:
    json.dump(label_mapping, outfile, indent=4)

print("Label mapping created successfully!")

