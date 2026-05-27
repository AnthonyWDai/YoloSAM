from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import torch


@dataclass
class SAMFinetuneConfig:
    device: str = "cuda"
    num_workers: int = 1
    sam_path: str = "pretrained/sam_vit_h_4b8939.pth"
    checkpoint_path: Optional[str] = None
    output_path: Optional[str] = None
    model_type: str = "vit_b"
    image_size: int = 1024
    
    # training
    batch_size: int = 2
    num_epochs: int = 100
    
    # loss Dice + BCE + KL divergence -> Dice = 1 - BCE - KL
    lambda_bce: float = 0.2 # for BCE loss (0.2)
    lambda_kl: float = 0.2 # for KL divergence (0.2)
    sigma: float = 1.0 # for soft label (KL divergence)
    
    # optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # wandb
    wandb_project: str = "SAM-finetune"
    wandb_name: str = "test"
    wandb_mode: str = "disabled"
    
    
    
@dataclass
class SAMDatasetConfig:
    dataset_path: str = "data/dataset"
    image_size: int = 1024
    
    # Agumentation
    percentiles: Tuple[float, float] = (0.1, 99.9)
    # rotation
    rotate_limit: float = 15
    rotate_prob: float = 0.5
    
    # scale
    scale_limit: float = 0.1
    scale_prob: float = 0.5
    
    # horizontal flip
    horizontal_flip_prob: float = 0.5
    
    # gamma
    gamma_prob: float = 0.7
    gamma_limit: Tuple[float, float] = (80, 120)
    
    # train or val
    train: bool = True
    
    # remove non-scar
    remove_nonscar: bool = True
    
    # prompt
    # YOLO prompt
    yolo_prompt: bool = False # if True, use yolo boxes as prompt
    yolo_model_path: str = "checkpoints/yolo11n.pt"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_imgsz: int = 640
    
    # box prompt
    box_prompt: bool = True
    enable_direction_aug: bool = True
    enable_size_aug: bool = True

    # point prompt
    point_prompt: bool = True
    num_points: int = 3
    point_prompt_types: List[str] = field(default_factory=lambda: ['positive'])
    
    # sample size (For testing)
    sample_size: int = 100

@dataclass
class YOLOConfig:
    # Model configuration
    model_type: str = "yolo11n"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None
    pretrained_path: Optional[str] = None
    output_path: Optional[str] = None
    
    # Dataset configuration
    dataset_path: str = "data/train"
    val_dataset_path: Optional[str] = None
    class_names: List[str] = field(default_factory=lambda: ['scar'])
    
    # Training parameters
    epochs: int = 300
    batch_size: int = 16
    image_size: int = 640
    patience: int = 50
    
    # Loss weights
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    
    # Detection parameters
    iou_threshold: float = 0.2
    conf_threshold: float = 0.15
    max_detections: int = 1
    
    # Augmentation parameters
    mosaic: float = 0.9
    mixup: float = 0.1
    copy_paste: float = 0.4
    scale: float = 0.5
    fliplr: float = 0.5
    flipud: float = 0.1
    degrees: float = 15.0
    translate: float = 0.3
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.3
    
    # Optimizer parameters
    learning_rate: float = 0.001
    final_lr_ratio: float = 0.0001
    warmup_epochs: int = 10
    weight_decay: float = 0.001
    
    # Training settings
    multi_scale: bool = True
    test_time_augmentation: bool = True
    save_period: int = 50
    
    # Project settings
    project_name: str = "yolo_training"
    experiment_name: str = "scar_detection"
    
    # Wandb settings
    wandb_project: str = "YOLO-training"
    wandb_name: str = "scar_detection"
    wandb_mode: str = "disabled"
    
@dataclass
class YoloSAMInferenceConfig:
    yolo_checkpoint_path: str = "checkpoints/yolo11n.pt"
    sam_checkpoint_path: str = "checkpoints/sam_vit_b_01ec64.pth"
    device: str = "cpu"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 1
    
    
