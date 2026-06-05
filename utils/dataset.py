import os
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

from utils.config import SAMDatasetConfig
from utils.prompt import BoxPromptGenerator, PointPromptGenerator
from utils.z_score_norm import PercentileNormalize


class SAMDataset(torch.utils.data.Dataset):
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')
    MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')

    def __init__(self, config: Union[Dict, SAMDatasetConfig]):
        self.config = config if isinstance(config, SAMDatasetConfig) else SAMDatasetConfig(**config)

        self.box_generator = BoxPromptGenerator(
            enable_direction_aug=self.config.enable_direction_aug,
            enable_size_aug=self.config.enable_size_aug,
            image_shape=(self.config.image_size, self.config.image_size)
        )
        self.point_generator = PointPromptGenerator(
            strategies=self.config.point_prompt_types,
            number_of_points=self.config.num_points
        )

        if self.config.train:
            self.train_transforms = A.Compose([
                A.RandomGamma(gamma_limit=self.config.gamma_limit, p=self.config.gamma_prob),
                A.Rotate(limit=self.config.rotate_limit, p=self.config.rotate_prob),
                A.RandomScale(scale_limit=self.config.scale_limit, p=self.config.scale_prob),
                A.HorizontalFlip(p=self.config.horizontal_flip_prob),
                A.Resize(self.config.image_size, self.config.image_size),
                PercentileNormalize(
                    lower_percentile=self.config.percentiles[0],
                    upper_percentile=self.config.percentiles[1]
                ),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.val_transforms = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                PercentileNormalize(
                    lower_percentile=self.config.percentiles[0],
                    upper_percentile=self.config.percentiles[1]
                ),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})

        self.samples: List[Dict[str, str]] = []
        self._load_dataset()

        if self.config.remove_nonscar:
            self._remove_nonscar()

        if self.config.yolo_prompt:
            self.yolo_model = YOLO(self.config.yolo_model_path)

    def _is_image_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.IMAGE_EXTENSIONS)

    def _is_mask_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.MASK_EXTENSIONS)

    def _get_stem(self, filename: str) -> str:
        return os.path.splitext(filename)[0]

    def _find_matching_mask(self, mask_dir: str, image_filename: str) -> Optional[str]:
        """
        Find a mask file with the same stem as image_filename in mask_dir,
        regardless of extension.
        Example:
            image: a.npy -> mask can be a.png / a.jpg / a.npy
        """
        image_stem = self._get_stem(image_filename)

        # First try exact filename
        exact_path = os.path.join(mask_dir, image_filename)
        if os.path.exists(exact_path):
            return exact_path

        # Then try same stem with any supported mask extension
        for ext in self.MASK_EXTENSIONS:
            candidate = os.path.join(mask_dir, image_stem + ext)
            if os.path.exists(candidate):
                return candidate

        return None

    def _load_npy(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            raise ValueError(f"Expected .npy file but got .npz-like object: {path}")
        return arr

    def _load_image(self, path: str) -> np.ndarray:
        """
        Load image as HWC numpy array.
        Supports standard image formats and .npy.

        Expected .npy image shapes:
        - H, W           -> converted to H, W, 3
        - H, W, 1        -> converted to H, W, 3
        - H, W, 3        -> kept
        - 1, H, W        -> converted to H, W, 3
        - 3, H, W        -> converted to H, W, 3
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            image = np.load(path)

            if image.ndim == 2:
                pass
            elif image.ndim == 3:
                if image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))

                if image.shape[-1] == 1:
                    image = image[..., 0]
                elif image.shape[-1] == 3:
                    image = image.astype(np.float32)
                    lo, hi = np.percentile(image, [1, 99])
                    if hi > lo:
                        image = np.clip(image, lo, hi)
                        image = (image - lo) / (hi - lo)
                    else:
                        image = np.zeros_like(image, dtype=np.float32)
                    return (image * 255).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported channel dimension in .npy image shape {image.shape} for file: {path}")
            else:
                raise ValueError(f"Unsupported .npy image shape {image.shape} for file: {path}")

            image = image.astype(np.float32)
            lo, hi = np.percentile(image, [1, 99])
            if hi > lo:
                image = np.clip(image, lo, hi)
                image = (image - lo) / (hi - lo)
            else:
                image = np.zeros_like(image, dtype=np.float32)

            image = (image * 255).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)
            return image

        # Standard image files
        return np.array(Image.open(path).convert('RGB'))

    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load mask as HW float32 binary array in {0,1}.
        Supports standard image formats and .npy.

        Expected .npy mask shapes:
        - H, W
        - H, W, 1
        - 1, H, W
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == '.npy':
            mask = self._load_npy(path)

            if mask.ndim == 3:
                if mask.shape[0] == 1 and mask.shape[-1] != 1:
                    mask = mask[0]
                elif mask.shape[-1] == 1:
                    mask = mask[..., 0]
                else:
                    raise ValueError(f"Unsupported .npy mask shape {mask.shape} for {path}")
            elif mask.ndim != 2:
                raise ValueError(f"Unsupported .npy mask ndim={mask.ndim} for {path}")

            mask = (mask > 0).astype(np.float32)
            return mask

        # Standard mask image files
        mask = np.array(Image.open(path).convert('L'))
        mask = (mask > 0).astype(np.float32)
        return mask

    def _load_dataset(self):
        """
        Load dataset from dataset_path.
        Supports two structures:
        1) Flat file structure:
            images/
                a.png
                b.npy
            masks/
                a.png
                b.npy

        2) Image-folder structure:
            images/
                case_001/
                    img1.png
                    img2.npy
                case_002/
                    img1.png
            masks/
                case_001/
                    img1.png
                    img2.npy
                case_002/
                    img1.png

        Matching is done by file stem, not necessarily exact extension.
        """
        image_dir = os.path.join(self.config.dataset_path, 'images')
        mask_dir = os.path.join(self.config.dataset_path, 'masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise RuntimeError(f"Dataset directories not found: {image_dir} or {mask_dir}")

        image_entries = sorted(os.listdir(image_dir))
        if not image_entries:
            raise RuntimeError(f"No entries found in image directory: {image_dir}")

        has_subdirs = any(os.path.isdir(os.path.join(image_dir, entry)) for entry in image_entries)
        has_files = any(
            os.path.isfile(os.path.join(image_dir, entry)) and self._is_image_file(entry)
            for entry in image_entries
        )

        if has_subdirs and has_files:
            raise RuntimeError(
                f"Mixed dataset structure detected in {image_dir}. "
                "Please use either all files or all subfolders."
            )

        samples = []

        if has_subdirs:
            for folder_name in image_entries:
                image_subdir = os.path.join(image_dir, folder_name)
                mask_subdir = os.path.join(mask_dir, folder_name)

                if not os.path.isdir(image_subdir):
                    continue
                if not os.path.isdir(mask_subdir):
                    print(f"Mask subfolder not found for image folder: {folder_name}")
                    continue

                for file_name in sorted(os.listdir(image_subdir)):
                    if not self._is_image_file(file_name):
                        continue

                    image_path = os.path.join(image_subdir, file_name)
                    mask_path = self._find_matching_mask(mask_subdir, file_name)

                    if mask_path is not None:
                        samples.append({
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'image_name': folder_name,
                            'file_name': file_name
                        })
                    else:
                        print(f"Mask not found for {folder_name}/{file_name}")

        elif has_files:
            for file_name in image_entries:
                if not self._is_image_file(file_name):
                    continue

                image_path = os.path.join(image_dir, file_name)
                mask_path = self._find_matching_mask(mask_dir, file_name)

                if mask_path is not None:
                    samples.append({
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'image_name': os.path.splitext(file_name)[0],
                        'file_name': file_name
                    })
                else:
                    print(f"Mask not found for image: {file_name}")

        else:
            raise RuntimeError(f"No valid image files or subfolders found in {image_dir}")

        if self.config.sample_size:
            sample_n = min(self.config.sample_size, len(samples))
            samples = random.sample(samples, sample_n)

        self.samples = samples
        print(f"Loaded {len(self.samples)} image-mask pairs")

    def _remove_nonscar(self):
        """
        Remove non-scar images from the dataset.
        If the mask is empty (sum of mask is less than 5), it is considered non-scar.
        """
        valid_samples = []
        removed_count = 0

        for sample in self.samples:
            mask = self._load_mask(sample['mask_path'])
            if mask.sum() >= 5:
                valid_samples.append(sample)
            else:
                removed_count += 1

        self.samples = valid_samples
        print(f"Removed {removed_count} empty masks")
        print(f"Loaded {len(self.samples)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image = self._load_image(sample['image_path'])
        mask = self._load_mask(sample['mask_path'])

        transforms = self.train_transforms if self.config.train else self.val_transforms

        for _ in range(5):
            transformed = transforms(image=image, mask=mask)
            image_t = transformed['image']
            mask_t = transformed['mask']
            mask_np = mask_t.cpu().numpy()
            if mask_np.sum() > 0:
                break
        else:
            image_t = transformed['image']
            mask_t = transformed['mask']
            mask_np = mask_t.cpu().numpy()

        mask_t = mask_t.unsqueeze(0)

        points_coords = None
        points_labels = None
        boxes = None

        if self.config.yolo_prompt:
            if isinstance(image_t, torch.Tensor):
                image_for_yolo = image_t.permute(1, 2, 0).cpu().numpy()
                if image_for_yolo.max() <= 1.0:
                    image_for_yolo = (image_for_yolo * 255).astype(np.uint8)
                else:
                    image_for_yolo = image_for_yolo.astype(np.uint8)
            else:
                image_for_yolo = image_t

            results = self.yolo_model.predict(
                image_for_yolo,
                conf=self.config.yolo_conf_threshold,
                iou=self.config.yolo_iou_threshold,
                imgsz=self.config.yolo_imgsz,
                device=self.config.device,
                verbose=False
            )

            if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes = torch.tensor(results[0].boxes.xyxy.cpu().numpy(), dtype=torch.float32)
            else:
                box = self.box_generator.generate(mask_np)
                boxes = torch.tensor(box, dtype=torch.float32)
        else:
            if self.config.point_prompt:
                points, labels = self.point_generator.generate(mask_np)
                points_coords = torch.tensor(points, dtype=torch.float32)
                points_labels = torch.tensor(labels, dtype=torch.float32)

            if self.config.box_prompt:
                box = self.box_generator.generate(mask_np)
                boxes = torch.tensor(box, dtype=torch.float32)

        return {
            'image': image_t.float(),
            'mask': mask_t.float(),
            'points_coords': points_coords,
            'points_labels': points_labels,
            'boxes': boxes,
            'image_name': sample['image_name'],
            'file_name': sample['file_name'],
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
        }


if __name__ == "__main__":
    config = SAMDatasetConfig(
        dataset_path='./sample_data/train',
        image_size=1024,
        point_prompt=True,
        box_prompt=True,
        num_points=3,
        train=True,
        remove_nonscar=True,
        yolo_prompt=True,
        yolo_model_path='runs/yolo_scar_detection2/weights/best.pt',
        point_prompt_types=['positive'],
        sample_size=10
    )

    dataset = SAMDataset(config)
    data = dataset[0]

    print(data['image'].shape)
    print(data['mask'].shape)
    print(data['boxes'].shape if data['boxes'] is not None else None)
    print(data['boxes'])
    print("image_name:", data['image_name'])
    print("file_name:", data['file_name'])
    print("image_path:", data['image_path'])
    print("mask_path:", data['mask_path'])