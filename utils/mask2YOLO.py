import cv2
import numpy as np

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional


class Mask2YOLOConverter:
    """Convert mask images to YOLO format labels."""
    
    def __init__(self, class_id: int = 0):
        self.class_id = class_id

    def mask_to_bboxes(
        self, 
        mask: np.ndarray, 
        min_area: int = 100
    ) -> List[Tuple[float, float, float, float]]:
        """
        Convert binary mask to YOLO format bounding boxes.

        Args:
            mask: Binary mask (0 and 255 or 0 and 1)
            min_area: Minimum area threshold for filtering small objects

        Returns:
            List of bounding boxes in YOLO format:
            [x_center, y_center, width, height] normalized to [0, 1]
        """
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        height, width = mask.shape

        # the function actually separates instance same classes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < min_area:
                continue

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            width_norm = w / width
            height_norm = h / height

            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))

            bboxes.append((x_center, y_center, width_norm, height_norm))

        return bboxes

    def convert_single_mask(self, mask_path: Path, output_path: Path, min_area: int = 100):
        """Convert a single mask file to YOLO label format."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return

        bboxes = self.mask_to_bboxes(mask, min_area)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                x_center, y_center, width, height = bbox
                f.write(
                    f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

    def _find_mask_files(self, mask_dir: Path) -> List[Path]:
        """Recursively find mask files in flat or nested folder structures."""
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        return sorted([p for p in mask_dir.rglob('*') if p.is_file() and p.suffix.lower() in exts])

    def convert_dataset_inplace(
        self,
        base_path: str,
        min_area: int = 100,
        splits: List[str] = ['train', 'val']
    ):
        """
        Convert masks to YOLO labels while preserving folder structure.

        Supports both:
        1) Flat structure:
           dataset/train/masks/image_001.png

        2) Nested structure:
           dataset/train/masks/cats/cat_1.png
           dataset/train/masks/dogs/dog_1.png

        Output labels are written to:
           dataset/train/labels/...
           dataset/val/labels/...

        preserving relative paths from masks/.
        """
        base_path = Path(base_path)

        for split in splits:
            print(f"Processing {split} split...")

            split_dir = base_path / split
            mask_dir = split_dir / 'masks'
            label_dir = split_dir / 'labels'

            if not mask_dir.exists():
                print(f"Warning: Mask directory {mask_dir} does not exist")
                continue

            mask_files = self._find_mask_files(mask_dir)

            print(f"Found {len(mask_files)} mask files in {split}")

            for mask_file in tqdm(mask_files, desc=f"Converting {split}"):
                # Preserve nested folder structure relative to masks/
                relative_path = mask_file.relative_to(mask_dir)
                label_file = (label_dir / relative_path).with_suffix('.txt')

                self.convert_single_mask(mask_file, label_file, min_area)

            print(f"Completed {split} split: {len(mask_files)} files processed")
            print(f"Labels saved to: {label_dir}")

        print("Dataset conversion completed!")
        print("Your dataset structure is now:")
        for split in splits:
            split_dir = base_path / split
            if split_dir.exists():
                print(f"  {split_dir}/")
                print(f"    images/")
                print(f"    masks/")
                print(f"    labels/  <- NEW (same subfolder structure as masks)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert segmentation masks to YOLO labels in-place."
    )

    parser.add_argument(
        "--base-path",
        type=str,
        default="./sample_data",
        help="Root dataset path containing split folders (default: ./sample_data)",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class ID to assign to all converted objects (default: 0)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=5,
        help="Minimum mask area threshold for keeping objects (default: 5)",
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process, e.g. --splits train val test (default: train val)",
    )

    return parser.parse_args()


def main():
    """Convert masks to YOLO labels in the same folder structure."""
    args = parse_args()

    converter = Mask2YOLOConverter(class_id=args.class_id)

    converter.convert_dataset_inplace(
        base_path=args.base_path,
        min_area=args.min_area,
        splits=args.splits,
    )


if __name__ == "__main__":
    main() 