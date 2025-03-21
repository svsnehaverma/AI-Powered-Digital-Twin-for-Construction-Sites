import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Directories
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2"
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
                    action_description = {0: 'Idle', 1: 'Swing Bucket', 2: 'Load Bucket', 3: 'Dump', 4: 'Move'}.get(action_type, 'Unknown')
                    activities.append((frame_start, frame_end, action_description))
    return activities

# Create a list to store activities and corresponding frame numbers
frame_activity_data = []  # Will store tuples of (frame_number, equipment, activity)

# Iterate over each folder in the frames directory
for folder_name in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder_name)
    if os.path.isdir(folder_path):  # Ensure it's a directory
        print(f"Processing folder: {folder_name}")
        
        # Check for corresponding label files in the Labels directory
        label_folder = os.path.join(labels_dir, folder_name)
        if not os.path.exists(label_folder):
            print(f"No label folder found for {folder_name}")
            continue
        
        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
        activities = {}
        
        # Load activities from all label files in the current folder
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            equipment = os.path.splitext(label_file)[0]  # Get equipment name (excavator, truck1, etc.)
            print(f"Loading activities for {equipment}")
            activities[equipment] = load_activities(label_path)

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

                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"Failed to load frame: {frame_path}")
                    continue

                # Match activity labels for each equipment in the frame
                for equipment, activity_list in activities.items():
                    for (start, end, action) in activity_list:
                        if start <= frame_number <= end:
                            frame_activity_data.append((frame_number, equipment, action))  # Save (frame_number, equipment, activity)

# Convert the activities to a format suitable for plotting
vehicles = set([data[1] for data in frame_activity_data])  # Get unique vehicles
activity_labels = ['Idle', 'Swing Bucket', 'Load Bucket', 'Dump', 'Move']  # Define activity labels

# Create a color map for activities
activity_map = {activity: idx for idx, activity in enumerate(activity_labels)}
color_map = plt.cm.get_cmap('tab10', len(activity_labels))  # Use a color map with sufficient colors

# Create a plot for each vehicle in a subplot
fig, axes = plt.subplots(len(vehicles), 1, figsize=(10, 6 * len(vehicles)))
if len(vehicles) == 1:
    axes = [axes]  # Ensure axes is iterable even for a single plot

for i, vehicle in enumerate(vehicles):
    # Get the data for the current vehicle
    vehicle_data = [(frame, activity) for (frame, equipment, activity) in frame_activity_data if equipment == vehicle]

    # Extract frame numbers and corresponding activities
    frames = [data[0] for data in vehicle_data]
    activities = [data[1] for data in vehicle_data]

    # Map the activities to numeric values for plotting purposes
    numeric_activities = [activity_map[activity] for activity in activities]

    # Plot the activity cycles of the vehicle
    axes[i].scatter(frames, numeric_activities, c=numeric_activities, cmap=color_map, s=20)
    
    # Identify when the activity switches (i.e., when the current activity differs from the previous one)
    for j in range(1, len(numeric_activities)):
        if numeric_activities[j] != numeric_activities[j - 1]:
            # Add a vertical line at the point of switch (activity change)
            axes[i].axvline(x=frames[j], color='gray', linestyle='--', alpha=0.5)

    axes[i].set_yticks(range(len(activity_labels)))
    axes[i].set_yticklabels(activity_labels)
    axes[i].set_xlabel('Frame Number')
    axes[i].set_ylabel('Activity')
    axes[i].set_title(f'{vehicle} Activity Over Time')
    axes[i].grid(True)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

