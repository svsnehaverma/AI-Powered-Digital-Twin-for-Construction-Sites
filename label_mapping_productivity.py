import os
import xml.etree.ElementTree as ET
import json
import matplotlib.pyplot as plt

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

# Function to calculate downtime ratio for equipment
def calculate_downtime_ratio(activities, total_frames):
    downtime = 0
    worktime = 0
    
    for (start, end, action) in activities:
        if action == 'Idle':
            downtime += (end - start + 1)
        else:
            worktime += (end - start + 1)
    
    # Avoid division by zero if no worktime
    if downtime + worktime == 0:
        return 0
    downtime_ratio = (downtime / (downtime + worktime)) * 100
    return downtime_ratio

# Main function to create label_mapping
def create_label_mapping(frames_dir, labels_dir, annotations_dir):
    label_mapping = {}

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
                        for equipment, activity_list in activities.items():
                            for (start, end, action) in activity_list:
                                if start <= frame_number <= end:
                                    frame_activities[equipment] = action
                                    break

                        # Get bounding boxes
                        bounding_boxes = get_bounding_boxes(xml_file, frame_number)

                        # Placeholder for productivity label (downtime ratio)
                        productivity_label = {}
                        for equipment, activity_list in activities.items():
                            downtime_ratio = calculate_downtime_ratio(activity_list, len(frame_activities))
                            productivity_label[equipment] = downtime_ratio

                        # Save to label_mapping
                        label_mapping[folder_name][frame_number] = {
                            'activities': frame_activities,
                            'bounding_boxes': bounding_boxes,
                            'productivity': productivity_label
                        }

    return label_mapping

# Generate the label mapping
label_mapping = create_label_mapping(frames_dir, labels_dir, annotations_dir)

# Save or print the label_mapping for verification
with open("label_mapping_downtime.json", "w") as outfile:
    json.dump(label_mapping, outfile, indent=4)

print("Label mapping created successfully!")

# Plot the downtime ratio for each equipment
# For this example, we'll plot the downtime ratio for the first folder and frame numbers
folder_name = list(label_mapping.keys())[0]  # First folder for demonstration
equipment_list = list(label_mapping[folder_name].values())[0]['productivity'].keys()

# Plotting the downtime ratios for each equipment
plt.figure(figsize=(10, 6))
for equipment in equipment_list:
    downtime_ratios = []
    frames = []
    for frame_number, data in label_mapping[folder_name].items():
        if equipment in data['productivity']:
            downtime_ratios.append(data['productivity'][equipment])
            frames.append(frame_number)
    
    plt.plot(frames, downtime_ratios, label=equipment)

plt.xlabel('Frame Number')
plt.ylabel('Downtime Ratio (%)')
plt.title('Downtime Ratio for Each Equipment')
plt.legend()
plt.grid(True)
plt.show()

