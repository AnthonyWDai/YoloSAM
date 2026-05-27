from ultralytics import YOLO
from pathlib import Path
import yaml
import os
import torch
import numpy as np
from typing import Optional, List, Tuple, Union

from utils.config import YOLOConfig

class YOLOModel:
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.device = config.device
        self.model = None
        self.dataset_yaml_path = None

        # Initialize model
        self.load_model()
        
        # Create dataset configuration only if dataset_path is provided (for training)
        if self.config.dataset_path is not None:
            self.create_dataset_yaml()
    
    def load_model(self):
        """Load YOLO model from pretrained weights or checkpoint."""
        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            print(f"Loading YOLO model from checkpoint: {self.config.checkpoint_path}")
            self.model = YOLO(self.config.checkpoint_path)
        else:
            print(f"Loading pretrained YOLO model: {self.config.model_type}")
            self.model = YOLO(f'{self.config.model_type}.pt')
    
    def create_dataset_yaml(self):
        """Create the dataset configuration file for YOLO training."""
        if self.config.dataset_path is None:
            print("No dataset path provided, skipping dataset YAML creation")
            return None
        
        # Use absolute paths to avoid relative path issues
        data_yaml = {
            'path': str(Path(self.config.dataset_path).parent.absolute()),
            'train': Path(self.config.dataset_path).name + '/images',
            'val': Path(self.config.val_dataset_path).name + '/images' if self.config.val_dataset_path else 'val/images',
            'names': {i: name for i, name in enumerate(self.config.class_names)},
            'nc': len(self.config.class_names)
        }
        
        self.dataset_yaml_path = Path(self.config.dataset_path).parent / 'dataset.yaml'
        with open(self.dataset_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"Created dataset configuration at: {self.dataset_yaml_path}")
        return str(self.dataset_yaml_path)
    
    def train(self, **kwargs):
        """Train the YOLO model with the configured parameters."""
        if not self.dataset_yaml_path:
            self.create_dataset_yaml()
        
        # Merge config with any additional kwargs
        train_config = {
            'data': str(self.dataset_yaml_path),
            'epochs': self.config.epochs,
            'imgsz': self.config.image_size,
            'batch': self.config.batch_size,
            'patience': self.config.patience,
            
            # Detection parameters
            'box': self.config.box_loss_gain,
            'cls': self.config.cls_loss_gain,
            'dfl': self.config.dfl_loss_gain,
            
            # Detection thresholds
            'iou': self.config.iou_threshold,
            'max_det': self.config.max_detections,
            
            # Augmentation parameters
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'copy_paste': self.config.copy_paste,
            'scale': self.config.scale,
            'fliplr': self.config.fliplr,
            'flipud': self.config.flipud,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            
            # Optimizer parameters
            'lr0': self.config.learning_rate,
            'lrf': self.config.final_lr_ratio,
            'warmup_epochs': self.config.warmup_epochs,
            'weight_decay': self.config.weight_decay,
            
            # Save settings
            'save_period': self.config.save_period,
            'project': self.config.project_name,
            'name': self.config.experiment_name,
            
            # Multi-scale training
            'multi_scale': self.config.multi_scale,
            
            # Device
            'device': self.config.device,
        }
        
        # Update with any additional kwargs
        train_config.update(kwargs)
        
        print("Starting YOLO training with configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        
        results = self.model.train(**train_config)
        return results
    
    def predict(self, source: Union[str, Path, np.ndarray], **kwargs):
        """Make predictions with the YOLO model."""
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")
        
        predict_config = {
            'conf': self.config.conf_threshold,
            'iou': self.config.iou_threshold,
            'max_det': self.config.max_detections,
            'augment': self.config.test_time_augmentation,
            'device': self.device,
            'verbose': False,
        }
        
        # Update with any additional kwargs
        predict_config.update(kwargs)
        
        results = self.model.predict(source=source, **predict_config)
        return results
    
    def validate(self, **kwargs):
        """Validate the YOLO model."""
        if not self.model:
            raise ValueError("Model not loaded. Please load a model first.")
        
        val_config = {
            'data': str(self.dataset_yaml_path) if self.dataset_yaml_path else None,
            'device': self.device,
        }
        
        # Update with any additional kwargs
        val_config.update(kwargs)
        
        results = self.model.val(**val_config)
        return results
    
    def save(self, path: Union[str, Path]):
        """Save the model to specified path."""
        if not self.model:
            raise ValueError("Model not loaded. Cannot save.")
        
        self.model.save(str(path))
        print(f"Model saved to: {path}")
    
    def export(self, format: str = 'onnx', **kwargs):
        """Export the model to different formats."""
        if not self.model:
            raise ValueError("Model not loaded. Cannot export.")
        
        return self.model.export(format=format, **kwargs)
    
    def get_model_info(self):
        """Get information about the loaded model."""
        if not self.model:
            return "No model loaded"
        
        return self.model.info()