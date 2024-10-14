'''
########## Testing for one files from the dataset #############
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Directories
frames_dir = "/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/frames_ce/"
annotations_file = "/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/DetectionTrackingAnnotations/150625.xml"

# File paths for activity labels
excavator_file = '/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/Labels/150625/excavator.txt'
truck1_file = '/Users/sneha/Library/CloudStorage/OneDrive-NewcastleUniversity/AI in Digital Twin/Dataset/Data/Data/Labels/150625/truck1.txt'

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

# Load activities for both pieces of equipment
activities = {
    "excavator": load_activities(excavator_file),
    "truck1": load_activities(truck1_file)
}

# Function to extract polygon and convert to bounding box from XML
def get_bounding_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bounding_boxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
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

            print(f"Bounding box for {label}: ({x_min}, {y_min}), ({x_max}, {y_max})")  # Debug print
            bounding_boxes.append({'label': label, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})
    return bounding_boxes

# Iterate over frame files and match activity labels
frames = []
for root, _, frame_files in os.walk(frames_dir):
    for frame_file in frame_files:
        if frame_file.endswith('.jpg'):
            print(f"Processing file: {frame_file}")
            try:
                frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])  # Extract frame number
            except ValueError:
                print(f"Unable to extract frame number from filename: {frame_file}")
                continue

            frame_path = os.path.join(root, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to load frame: {frame_path}")
                continue
            print(f"Successfully loaded frame: {frame_path}")

            # Match activity labels
            frame_activities = {}
            for equipment, activity_list in activities.items():
                for (start, end, action) in activity_list:
                    if start <= frame_number <= end:
                        frame_activities[equipment] = action
                        print(f"Matched activity for equipment {equipment}: {action} (Start: {start}, End: {end})")
                        break

            # Use the XML file 150626.xml to extract bounding boxes
            bounding_boxes = get_bounding_boxes(annotations_file)
            print(f"Bounding boxes for frame {frame_number}: {bounding_boxes}")  # Debug bounding boxes

            frames.append({'frame': frame, 'filename': frame_file, 'activities': frame_activities, 'annotations': bounding_boxes})

# Plot the loaded frames with custom bounding boxes and labels
print(f"Total frames loaded: {len(frames)}")
for frame_data in frames[:5]:  # Plot only the first 5 frames for brevity
    frame = frame_data['frame']
    annotations = frame_data['annotations']
    print(f"Annotations: {annotations}")  # Debug annotations
    activities = frame_data['activities']
    print(f"Activities: {activities}")  # Debug activities
    filename = frame_data.get('filename', 'N/A')

    if frame is not None:
        # Draw bounding boxes using OpenCV for better control
        for annotation in annotations:
            x_min, y_min, x_max, y_max, label = annotation['x_min'], annotation['y_min'], annotation['x_max'], annotation['y_max'], annotation['label']
            
            # Set custom color for different labels
            color = (0, 255, 0) if label == 'excavator' else (255, 0, 0)  # Green for excavator, Blue for truck1
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)  # Draw bounding box with OpenCV
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Label the bounding box

        # Convert frame from BGR to RGB for displaying with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the frame with bounding boxes
        plt.figure(figsize=(10, 6))
        plt.imshow(frame_rgb)

        # Print activity descriptions inside the plot
        for equipment, action in activities.items():
            plt.text(10, 30 if equipment == 'excavator' else 60, f"{equipment.capitalize()}: {action}", fontsize=15, color='black', backgroundcolor='white')

        plt.title(f"Filename: {filename}")
        
        # Show the plot
        plt.show()
'''       
'''
# Plot the loaded frames with bounding boxes and labels
print(f"Total frames loaded: {len(frames)}")
for frame_data in frames[:5]:  # Plot only the first 5 frames for brevity
    frame = frame_data['frame']
    annotations = frame_data['annotations']
    activities = frame_data['activities']
    filename = frame_data.get('filename', 'N/A')

    if frame is not None:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for annotation in annotations:
            x, y = annotation['x'], annotation['y']
            plt.gca().add_patch(plt.Rectangle((x, y), 30, 30, fill=False, edgecolor='red', linewidth=2))  # Draw bounding box as a rectangle
        activity_text = ', '.join([f"{key}: {value}" for key, value in activities.items()])
        plt.title(f"Filename: {filename}\nFrame activities: {activity_text}")
        plt.show()
'''

##########################################################################################################
#####################################  Load All the Dataset ##############################################
##########################################################################################################
import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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
        frames = []
        for frame_file in os.listdir(folder_path):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(folder_path, frame_file)
                print(f"Processing frame: {frame_file}")
                
                try:
                    frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])  # Extract frame number
                except ValueError:
                    print(f"Unable to extract frame number from filename: {frame_file}")
                    continue

                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Failed to load frame: {frame_path}")
                    continue
                print(f"Successfully loaded frame: {frame_path}")

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

                # Filter bounding boxes and ensure they match the valid labels
                filtered_bounding_boxes = []
                added_labels = set()
                truck_counter = 1  # For sequential truck labels

                for bbox in bounding_boxes:
                    label = bbox['label']

                    # Assign truck labels sequentially if the label is "truck"
                    if label == 'truck' and truck_counter <= 3:
                        specific_label = f"truck{truck_counter}"
                        truck_counter += 1
                    else:
                        specific_label = label

                    # Avoid duplicates
                    if specific_label not in added_labels:
                        bbox['label'] = specific_label
                        filtered_bounding_boxes.append(bbox)
                        added_labels.add(specific_label)

                # Ensure the number of bounding boxes matches the number of activities
                activity_count = len(frame_activities)
                if len(filtered_bounding_boxes) > activity_count:
                    filtered_bounding_boxes = filtered_bounding_boxes[:activity_count]

                # Print the bounding boxes for the frame
                print(f"Bounding boxes for frame {frame_file}:")
                for annotation in filtered_bounding_boxes:
                    label = annotation['label']
                    print(f"  Bounding box for {label}: ({annotation['x_min']}, {annotation['y_min']}) to ({annotation['x_max']}, {annotation['y_max']})")

                frames.append({'frame': frame, 'filename': frame_file, 'activities': frame_activities, 'annotations': filtered_bounding_boxes})

        # Plot the first 5 frames with bounding boxes and labels
        print(f"Total frames processed for {folder_name}: {len(frames)}")
        for frame_data in frames[:5]:  # Plot only the first 5 frames for brevity
            frame = frame_data['frame']
            annotations = frame_data['annotations']
            activities = frame_data['activities']
            filename = frame_data.get('filename', 'N/A')

            if frame is not None:
                # Draw bounding boxes using OpenCV for better control
                for annotation in annotations:
                    x_min, y_min, x_max, y_max, label = annotation['x_min'], annotation['y_min'], annotation['x_max'], annotation['y_max'], annotation['label']

                    # Set custom color for different labels
                    color = (0, 255, 0) if label == 'excavator' else (255, 0, 0)  # Green for excavator, Blue for truck1, truck2, truck3
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Convert frame from BGR to RGB for displaying with matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Show the frame with bounding boxes
                plt.figure(figsize=(10, 6))
                plt.imshow(frame_rgb)

                # Print activity descriptions inside the plot
                for equipment, action in activities.items():
                    plt.text(10, 30 if equipment == 'excavator' else 60, f"{equipment.capitalize()}: {action}", fontsize=15, color='black', backgroundcolor='white')

                plt.title(f"Filename: {filename}")
                plt.show()
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
        frames = []
        for frame_file in os.listdir(folder_path):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(folder_path, frame_file)
                print(f"Processing frame: {frame_file}")
                
                try:
                    frame_number = int(frame_file.split('_')[-1].split('.')[0][-5:])  # Extract frame number
                except ValueError:
                    print(f"Unable to extract frame number from filename: {frame_file}")
                    continue

                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Failed to load frame: {frame_path}")
                    continue
                print(f"Successfully loaded frame: {frame_path}")

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

                # Filter bounding boxes and ensure they match the valid labels
                filtered_bounding_boxes = []
                added_labels = set()
                truck_counter = 1  # For sequential truck labels

                for bbox in bounding_boxes:
                    label = bbox['label']

                    # Assign truck labels sequentially if the label is "truck"
                    if label == 'truck' and truck_counter <= 3:
                        specific_label = f"truck{truck_counter}"
                        truck_counter += 1
                    else:
                        specific_label = label

                    # Avoid duplicates
                    if specific_label not in added_labels:
                        bbox['label'] = specific_label
                        filtered_bounding_boxes.append(bbox)
                        added_labels.add(specific_label)

                # Ensure the number of bounding boxes matches the number of activities
                activity_count = len(frame_activities)
                if len(filtered_bounding_boxes) > activity_count:
                    filtered_bounding_boxes = filtered_bounding_boxes[:activity_count]

                # Print the bounding boxes for the frame
                print(f"Bounding boxes for frame {frame_file}:")
                for annotation in filtered_bounding_boxes:
                    label = annotation['label']
                    print(f"  Bounding box for {label}: ({annotation['x_min']}, {annotation['y_min']}) to ({annotation['x_max']}, {annotation['y_max']})")

                frames.append({'frame': frame, 'filename': frame_file, 'activities': frame_activities, 'annotations': filtered_bounding_boxes})

        # Plot the first 5 frames with bounding boxes and labels
        print(f"Total frames processed for {folder_name}: {len(frames)}")
        for frame_data in frames[:5]:  # Plot only the first 5 frames for brevity
            frame = frame_data['frame']
            annotations = frame_data['annotations']
            activities = frame_data['activities']
            filename = frame_data.get('filename', 'N/A')

            if frame is not None:
                # Draw bounding boxes using OpenCV for better control
                for annotation in annotations:
                    x_min, y_min, x_max, y_max, label = annotation['x_min'], annotation['y_min'], annotation['x_max'], annotation['y_max'], annotation['label']

                    # Set custom color for different labels
                    color = (0, 255, 0) if label == 'excavator' else (255, 0, 0)  # Green for excavator, Blue for truck1, truck2, truck3
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Convert frame from BGR to RGB for displaying with matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Show the frame with bounding boxes
                plt.figure(figsize=(10, 6))
                plt.imshow(frame_rgb)

                # Print activity descriptions inside the plot
                for equipment, action in activities.items():
                    plt.text(10, 30 if equipment == 'excavator' else 60, f"{equipment.capitalize()}: {action}", fontsize=15, color='black', backgroundcolor='white')

                plt.title(f"Filename: {filename}")
                plt.show()
