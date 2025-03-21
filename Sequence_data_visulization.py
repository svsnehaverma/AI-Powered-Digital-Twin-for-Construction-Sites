import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import cv2

# File path to the processed data
test_csv_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Sequence_data_split/Short_data/test_sequences.csv"
frames_dir = "/home/campus.ncl.ac.uk/nsv53/Sneha/AI in Digital Twin/Dataset/Data/frames_ce_2/"
# Load the processed data
df = pd.read_csv(test_csv_path)

# Convert string representations back to Python objects
df['frames'] = df['frames'].apply(ast.literal_eval)  # List of frame paths
df['activities'] = df['activities'].apply(ast.literal_eval)  # List of activities
df['bounding_boxes'] = df['bounding_boxes'].apply(ast.literal_eval)  # List of bounding boxes

# Visualization Functions

def visualize_frame(frame_file, bounding_boxes, frame_dir):
    """Visualize a single frame with bounding boxes."""
    frame_path = os.path.join(frame_dir, frame_file)
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    for bbox in bounding_boxes:
        rect = patches.Rectangle(
            (bbox['x_min'], bbox['y_min']),
            bbox['x_max'] - bbox['x_min'],
            bbox['y_max'] - bbox['y_min'],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            bbox['x_min'], bbox['y_min'] - 10,
            bbox['label'],
            color='red',
            fontsize=12,
            bbox=dict(facecolor='yellow', alpha=0.5)
        )

    plt.axis('off')
    plt.show()

def visualize_sequence(sequence, frame_dir):
    """Visualize sliding window sequence with bounding boxes and activities."""
    frames = sequence['frames']
    bounding_boxes = sequence['bounding_boxes']
    activities = sequence['activities']

    n = len(frames)
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    
    for i, frame_file in enumerate(frames):
        frame_path = os.path.join(frame_dir, frame_file)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f"Frame: {frame_file}")

        # Draw bounding boxes
        for bbox in bounding_boxes:
            if bbox.get('frame') == frame_file:  # Match bounding boxes to the frame
                rect = patches.Rectangle(
                    (bbox['x_min'], bbox['y_min']),
                    bbox['x_max'] - bbox['x_min'],
                    bbox['y_max'] - bbox['y_min'],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                axs[i].add_patch(rect)

    plt.suptitle(f"Activities: {', '.join(activities)}")
    plt.tight_layout()
    plt.show()

def plot_activity_histogram(seq_df):
    """Plot a histogram of activity frequencies."""
    all_activities = [activity for seq in seq_df['activities'] for activity in seq]
    activity_counts = Counter(all_activities)

    plt.bar(activity_counts.keys(), activity_counts.values(), color='skyblue')
    plt.xlabel('Activity')
    plt.ylabel('Frequency')
    plt.title('Activity Frequency in Dataset')
    plt.xticks(rotation=45)
    plt.show()

def plot_activity_timeline(seq_df):
    """Plot a timeline of activities across sequences."""
    timeline_data = []
    for idx, row in seq_df.iterrows():
        for activity in row['activities']:
            timeline_data.append({'sequence': idx, 'activity': activity})

    timeline_df = pd.DataFrame(timeline_data)

    plt.figure(figsize=(12, 6))
    for activity in timeline_df['activity'].unique():
        activity_data = timeline_df[timeline_df['activity'] == activity]
        plt.scatter(activity_data['sequence'], [activity] * len(activity_data), label=activity)

    plt.xlabel('Sequence Index')
    plt.ylabel('Activity')
    plt.title('Activity Timeline Across Sequences')
    plt.legend()
    plt.show()

# Example usage of visualization functions

# Visualize a single frame
frame_example = df.iloc[0]  # Pick the first sequence
first_frame = frame_example['frames'][0]  # Get the first frame in the sequence
bounding_boxes = [bbox for bbox in frame_example['bounding_boxes'] if bbox.get('frame') == first_frame]
visualize_frame(first_frame, bounding_boxes, frames_dir)

# Visualize a sliding window sequence
sequence_example = df.iloc[0].to_dict()  # Pick the first sequence
visualize_sequence(sequence_example, frames_dir)

# Plot activity histogram
plot_activity_histogram(df)

# Plot activity timeline
plot_activity_timeline(df)

