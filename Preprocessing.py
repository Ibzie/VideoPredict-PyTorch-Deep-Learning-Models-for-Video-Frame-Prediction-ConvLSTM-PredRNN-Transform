import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

CATEGORIES = ['YoYo','Typing','Punch','PizzaTossing','GolfSwing']
FRAME_SIZE = (64, 64)
USE_GRAYSCALE = True
BASE_PATH = 'ucf101'
TRAIN_PATH = os.path.join(BASE_PATH, 'train')
TEST_PATH = os.path.join(BASE_PATH, 'test')
OUTPUT_PATH = 'processed_data'
JPEG_QUALITY = 95


def create_output_dir(base_output_path):
    """Create necessary directories for saving processed frames."""
    for split in ['train', 'test']:
        for category in CATEGORIES:
            path = os.path.join(base_output_path, split, category)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


def process_video(video_path, output_path, video_name):
    """Process a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Resize frame
            frame = cv2.resize(frame, FRAME_SIZE)

            # Convert to grayscale if specified
            if USE_GRAYSCALE:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save individual frame as JPEG
            frame_filename = f"{video_name}_frame_{frame_count:03d}.jpg"
            frame_path = os.path.join(output_path, frame_filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            frame_count += 1

        except Exception as e:
            print(f"Error processing frame {frame_count} from video {video_path}: {str(e)}")
            continue

    cap.release()
    return frame_count


def process_dataset_split(split_path, output_base_path, split_name):
    """Process all videos in a dataset split (train or test)."""
    total_videos = 0
    total_frames = 0

    # Print all categories being processed
    print(f"\nProcessing categories: {', '.join(CATEGORIES)}")

    for category in CATEGORIES:
        category_path = os.path.join(split_path, category)
        output_path = os.path.join(output_base_path, split_name, category)

        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} not found")
            continue

        videos = [f for f in os.listdir(category_path) if f.endswith(('.avi', '.mp4'))]

        if not videos:
            print(f"Warning: No videos found in {category_path}")
            continue

        print(f"\nProcessing {len(videos)} videos for {category} in {split_name} split")

        category_videos = 0
        category_frames = 0

        for video in tqdm(videos, desc=f"Processing {category}"):
            video_path = os.path.join(category_path, video)
            video_name = os.path.split(video)[1].split('.')[0]
            frames = process_video(video_path, output_path, video_name)

            if frames > 0:
                category_videos += 1
                category_frames += frames

        total_videos += category_videos
        total_frames += category_frames

        print(f"Completed {category}: {category_videos} videos, {category_frames} frames")

    return total_videos, total_frames


def verify_data(output_path):
    """Verify processed data and display sample frames."""
    print("\nVerifying processed data:")

    for split in ['train', 'test']:
        print(f"\n{split.upper()} Split:")
        split_path = os.path.join(output_path, split)

        for category in CATEGORIES:
            category_path = os.path.join(split_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: Category path {category_path} not found")
                continue

            frames = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
            print(f"{category}: {len(frames)} frames")

            if frames:
                try:
                    sample_frame = cv2.imread(os.path.join(category_path, frames[0]))
                    if sample_frame is None:
                        print(f"Warning: Could not read sample frame for {category}")
                        continue

                    plt.figure(figsize=(5, 5))
                    if USE_GRAYSCALE:
                        plt.imshow(sample_frame, cmap='gray')
                    else:
                        sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                        plt.imshow(sample_frame)
                    plt.title(f"{category} - Sample Frame")
                    plt.axis('off')
                    plt.show()
                    plt.close()

                except Exception as e:
                    print(f"Error displaying sample frame for {category}: {str(e)}")


def main():
    # Verify categories are set
    if not CATEGORIES:
        raise ValueError("Please specify the categories in the configuration section")

    # Print configuration
    print("Starting video preprocessing with configuration:")
    print(f"Categories: {CATEGORIES}")
    print(f"Frame size: {FRAME_SIZE}")
    print(f"Grayscale: {USE_GRAYSCALE}")
    print(f"Base path: {BASE_PATH}")

    # Create output directories
    create_output_dir(OUTPUT_PATH)

    # Process training data
    print("\nProcessing training data...")
    train_videos, train_frames = process_dataset_split(TRAIN_PATH, OUTPUT_PATH, 'train')

    # Process test data
    print("\nProcessing test data...")
    test_videos, test_frames = process_dataset_split(TEST_PATH, OUTPUT_PATH, 'test')

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Training: {train_videos} videos, {train_frames} frames")
    print(f"Test: {test_videos} videos, {test_frames} frames")

    verify_data(OUTPUT_PATH)


if __name__ == "__main__":
    main()