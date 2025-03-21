import os
import matplotlib.pyplot as plt

# Function to load activities from file
def load_activities(file_path):
    activities = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 3:
                    frame_start, frame_end, action_type = map(int, parts)
                    activities.append((frame_start, frame_end, action_type))
    return activities

# Paths (update as per your system)
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce"
labels_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/Labels"

# Initialize data storage
vehicle_activity_data = {}  # {vehicle: [(start_frame, end_frame, action_type)]}

# Process directories
for folder_name in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, folder_name)
    if os.path.isdir(folder_path):
        label_folder = os.path.join(labels_dir, folder_name)
        if not os.path.exists(label_folder):
            continue

        label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
        for label_file in label_files:
            label_path = os.path.join(label_folder, label_file)
            equipment = os.path.splitext(label_file)[0]  # Extract equipment name
            activities = load_activities(label_path)
            if equipment not in vehicle_activity_data:
                vehicle_activity_data[equipment] = []
            vehicle_activity_data[equipment].extend(activities)

# Define activity labels and colors
activity_labels = {0: 'Idle', 1: 'Swing Bucket', 2: 'Load Bucket', 3: 'Dump', 4: 'Move'}
activity_colors = {0: 'lightcoral', 1: 'maroon', 2: 'mistyrose', 3: 'coral', 4: 'darkorange'}

# Plot all vehicles in a single figure
plt.figure(figsize=(12, len(vehicle_activity_data) * 2))

# Loop through each vehicle to create stacked bar plots
for idx, (vehicle, activities) in enumerate(vehicle_activity_data.items()):
    for start, end, action in activities:
        plt.barh(idx, end - start + 1, left=start, color=activity_colors[action], edgecolor="black", label=activity_labels[action])

# Customize the plot
plt.yticks(range(len(vehicle_activity_data)), list(vehicle_activity_data.keys()))
plt.xlabel('Time (Frames)', fontsize=12)
plt.ylabel('Vehicles', fontsize=12)
plt.title('Vehicle Activities Over Time (153004)', fontsize=14, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Create a legend
handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in activity_colors.items()]
plt.legend(handles, activity_labels.values(), title="Activities", loc="upper right", fontsize=10, title_fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

