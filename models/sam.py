import torch
from segment_anything import sam_model_registry
import torch.nn.functional as F
import torch.nn as nn
import os
from typing import Tuple, Optional, Dict

from utils.config import SAMFinetuneConfig


class SAMModel(nn.Module):
    def __init__(self, config: SAMFinetuneConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.load_model()
        
        self.image_encoder = self.model.image_encoder.to(self.device)
        self.mask_decoder = self.model.mask_decoder.to(self.device)
        self.prompt_encoder = self.model.prompt_encoder.to(self.device)

        # Freeze prompt encoder
        if config.freeze >= 1:
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        # Freeze image encoder
        if config.freeze >= 2:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # Train mask decoder
        if config.freeze >= 3:
            for p in self.mask_decoder.iou_token.parameters():
                p.requires_grad = False

            for p in self.mask_decoder.mask_tokens.parameters():
                p.requires_grad = False

        if config.freeze >= 4:
            for p in self.mask_decoder.transformer.parameters():
                p.requires_grad = False
        
        # Free memory
        del self.model
        torch.cuda.empty_cache()
        
    def load_model(self):
        """Load SAM model from checkpoint."""
        if self.config.sam_path is None or not os.path.exists(self.config.sam_path):
            print(f"SAM model checkpoint not found at {self.config.sam_path}, Please run `python download_sam.py` to download the base model.")
            
        sam = sam_model_registry[self.config.model_type](checkpoint=self.config.sam_path)
        print("Load SAM model from ", self.config.sam_path)
        sam.to(self.device)
        if self.config.checkpoint_path:
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device, weights_only=False)
            sam.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded sam checkpoint")
        return sam
    
    def forward_one_image(
        self, 
        image: torch.Tensor,
        bounding_box: Optional[torch.Tensor] = None,
        points: Optional[Dict[str, torch.Tensor]] = None,
        is_train: bool = True
    ) -> torch.Tensor:
        """Forward pass for SAM model."""
        if is_train:
            self.train()
        else:
            self.eval()
            
        image = image.to(self.device)
        image_size = image.shape[2:]
        
        image_embedding = self.image_encoder(image)
            
        # Prepare prompts 
        box = self._prepare_box(bounding_box) if bounding_box is not None else None
        pts = self._prepare_points(points) if points is not None else None
        # print("point_coords.shape: ", pts[0].shape)
        # print("point_labels.shape: ", pts[1].shape)
        # print("box.shape: ", box.shape)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=pts,
            boxes=box,
            masks=None,
        )
            
        low_res_masks, iou_prediction = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        high_res_masks = F.interpolate(
            low_res_masks,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        return high_res_masks, iou_prediction
    
    def _prepare_box(self, bounding_box: torch.Tensor) -> torch.Tensor:
        """Prepare bounding box for SAM model."""
        # Add batch dimension if not present
        if len(bounding_box.shape) == 1:
            bounding_box = bounding_box.unsqueeze(0)  # [4] -> [1, 4]
        if len(bounding_box.shape) == 2:
            bounding_box = bounding_box.unsqueeze(1)  # [1, 4] -> [1, 1, 4]
        return bounding_box.to(self.device)
    
    def _prepare_points(self, points: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare points for SAM model."""
        point_coords = points['coords'].to(self.device)
        point_labels = points['labels'].to(self.device)
        
        # Add batch dimension if not present
        if len(point_coords.shape) == 2:
            point_coords = point_coords.unsqueeze(0)  # [3, 2] -> [1, 3, 2]
        if len(point_labels.shape) == 1:
            point_labels = point_labels.unsqueeze(0)  # [3] -> [1, 3]
        
        return (point_coords, point_labels)
    
