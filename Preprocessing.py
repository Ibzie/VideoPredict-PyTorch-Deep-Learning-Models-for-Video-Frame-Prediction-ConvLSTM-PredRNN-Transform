import argparse
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import logging

# Constants
CATEGORIES = ['YoYo', 'Typing', 'Punch', 'PizzaTossing', 'GolfSwing']
FRAME_SIZE = (64, 64)

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Handle video preprocessing tasks."""
    
    def __init__(self, frame_size=FRAME_SIZE, use_grayscale=True, jpeg_quality=95):
        self.frame_size = frame_size
        self.use_grayscale = use_grayscale
        self.jpeg_quality = jpeg_quality

    def verify_data_structure(self, base_path):
        """Verify UCF101 data structure exists correctly."""
        base_path = Path(base_path)
        
        for split in ['train', 'test']:
            split_path = base_path / split
            if not split_path.exists():
                logger.error(f"Missing {split} directory at {split_path}")
                return False
                
            for category in CATEGORIES:
                category_path = split_path / category
                if not category_path.exists():
                    logger.error(f"Missing category directory: {category_path}")
                    return False
                
                videos = list(category_path.glob('*.avi')) + list(category_path.glob('*.mp4'))
                if not videos:
                    logger.error(f"No videos found in {category_path}")
                    return False
                
        return True

    def process_video(self, video_path, output_dir, video_name, frame_gap=7):
        """Process a single video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return 0

        frame_count = 0
        save_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_gap == 0:
                try:
                    frame = cv2.resize(frame, self.frame_size)
                    if self.use_grayscale:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    frame_filename = f"{video_name}_frame_{save_count:03d}.jpg"
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    save_count += 1

                except Exception as e:
                    logger.error(f"Error processing frame {frame_count} from {video_path}: {str(e)}")

            frame_count += 1

        cap.release()
        return save_count

    def process_dataset(self, input_base_path, output_base_path):
        """Process the entire UCF101 dataset maintaining category structure."""
        input_base_path = Path(input_base_path)
        output_base_path = Path(output_base_path)

        if not self.verify_data_structure(input_base_path):
            raise ValueError("Invalid data directory structure")

        for split in ['train', 'test']:
            for category in CATEGORIES:
                output_dir = output_base_path / split / category
                output_dir.mkdir(parents=True, exist_ok=True)

        total_stats = {'videos': 0, 'frames': 0}

        for split in ['train', 'test']:
            logger.info(f"\nProcessing {split} split...")
            split_stats = {'videos': 0, 'frames': 0}
            
            for category in CATEGORIES:
                input_dir = input_base_path / split / category
                output_dir = output_base_path / split / category
                
                videos = list(input_dir.glob('*.avi')) + list(input_dir.glob('*.mp4'))
                logger.info(f"\nProcessing {len(videos)} videos for {category}")
                
                for video_path in tqdm(videos, desc=f"{split}/{category}"):
                    n_frames = self.process_video(
                        video_path,
                        output_dir,
                        video_path.stem
                    )
                    
                    if n_frames > 0:
                        split_stats['videos'] += 1
                        split_stats['frames'] += n_frames

            logger.info(f"\n{split.capitalize()} Split Stats:")
            logger.info(f"Processed {split_stats['videos']} videos")
            logger.info(f"Generated {split_stats['frames']} frames")
            
            total_stats['videos'] += split_stats['videos']
            total_stats['frames'] += split_stats['frames']

        return total_stats

def main():
    parser = argparse.ArgumentParser(description='Preprocess UCF101 video dataset')
    parser.add_argument('--input-dir', type=str, default='ucf101',
                      help='Input directory containing UCF101 dataset')
    parser.add_argument('--output-dir', type=str, default='processed_data',
                      help='Output directory for processed frames')
    parser.add_argument('--grayscale', action='store_true',
                      help='Convert frames to grayscale')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting video preprocessing...")
    logger.info(f"Input path: {args.input_dir}")
    logger.info(f"Output path: {args.output_dir}")
    logger.info(f"Categories: {CATEGORIES}")
    logger.info(f"Frame size: {FRAME_SIZE}")
    logger.info(f"Grayscale: {args.grayscale}")

    preprocessor = VideoPreprocessor(
        frame_size=FRAME_SIZE,
        use_grayscale=args.grayscale
    )

    try:
        stats = preprocessor.process_dataset(args.input_dir, args.output_dir)
        logger.info("\nPreprocessing Complete!")
        logger.info(f"Total videos processed: {stats['videos']}")
        logger.info(f"Total frames generated: {stats['frames']}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()