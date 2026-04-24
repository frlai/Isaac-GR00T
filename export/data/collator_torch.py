"""
PyTorch-traceable collator for GR00T that replaces PIL-based image processing.

This collator is a full replacement for Gr00tN1d6DataCollator.
It handles torch tensor images and produces the same output format.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Any, Dict, List, Literal
from transformers import BatchFeature

from .image_processing_torch import (
    smart_resize,
    IMAGE_MEAN,
    IMAGE_STD,
)


def build_processor(model_name: str, transformers_loading_kwargs: dict):
    """Build the VLM processor (same as original)."""
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import build_processor as _build_processor
    return _build_processor(model_name, transformers_loading_kwargs)


class Gr00tN1d6DataCollatorTorch:
    """
    PyTorch-traceable data collator for GR00T N1D6.
    
    Full replacement for Gr00tN1d6DataCollator that:
    1. Takes images as torch tensors (H, W, C) uint8
    2. Processes images using pure PyTorch operations
    3. Gets tokenization from the underlying VLM processor
    4. Returns same BatchFeature format as original
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: Literal["eagle"] = "eagle",
        transformers_loading_kwargs: dict = {},
    ):
        """
        Args:
            model_name: HuggingFace model name for VLM processor
            model_type: Type of VLM backbone ("eagle")
            transformers_loading_kwargs: Kwargs for loading transformers
        """
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        
    def preprocess_image_torch(
        self,
        image: Tensor,
        target_height: int,
        target_width: int,
    ) -> Tensor:
        """
        Preprocess a single image using PyTorch operations.
        
        Replicates the Eagle image processor pipeline:
        1. Resize to target dimensions (bicubic interpolation)
        2. Rescale from [0, 255] to [0, 1]
        3. Normalize using mean=0.5, std=0.5 to get [-1, 1]
        
        Args:
            image: Input tensor (H, W, C) as uint8 [0, 255]
            target_height: Target height
            target_width: Target width
            
        Returns:
            Processed tensor (C, H, W) as float32 in [-1, 1]
        """
        # (H, W, C) -> (C, H, W)
        image = image.permute(2, 0, 1)
        
        # Add batch dim for interpolate
        image = image.unsqueeze(0).to(torch.float32)
        
        # Resize with bicubic interpolation (no antialias to match Eagle image processor)
        image = F.interpolate(
            image,
            size=(target_height, target_width),
            mode='bicubic',
            align_corners=False,
        )
        
        # Round and clamp to [0, 255] to simulate PIL uint8 behavior
        image = image.round().clamp(0, 255)
        
        # Remove batch dim
        image = image.squeeze(0)
        
        # Rescale: [0, 255] -> [0, 1]
        image = image / 255.0
        
        # Normalize: [0, 1] -> [-1, 1]
        image = (image - IMAGE_MEAN) / IMAGE_STD
        
        return image
    
    def process_images_torch(
        self,
        images: List[Tensor],
    ) -> tuple[List[Tensor], List[Tensor]]:
        """
        Process a list of images using PyTorch operations.
        
        Args:
            images: List of image tensors, each (H, W, C) as uint8
            
        Returns:
            Tuple of:
                - List of processed tensors, each (C, H, W) as float32 in [-1, 1]
                - List of size tensors, each [H, W]
        """
        processed = []
        sizes = []
        
        for img in images:
            # img is (H, W, C)
            h, w = img.shape[0], img.shape[1]
            
            # Compute target dimensions using smart_resize
            target_h, target_w = smart_resize(h, w)
            
            processed.append(self.preprocess_image_torch(img, target_h, target_w))
            sizes.append(torch.tensor([target_h, target_w]))
        
        return processed, sizes
    
    def extract_images_from_conversation(
        self,
        conversation: List[Dict],
    ) -> List[Tensor]:
        """
        Extract image tensors from conversation structure.
        
        Args:
            conversation: List of message dicts with 'content' containing image items
            
        Returns:
            List of image tensors
        """
        images = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item["image"]
                    images.append(img)
        return images
    
    def __call__(self, features: List[Dict[str, Any]]) -> BatchFeature:
        """
        Collate features into a batch.
        
        Args:
            features: List of processed inputs from processor.
                Each contains: vlm_content, state, embodiment_id, etc.
                Images in vlm_content should be torch tensors (H, W, C) uint8.
                
        Returns:
            BatchFeature with:
                - input_ids: tokenized text
                - attention_mask: attention mask for tokens
                - pixel_values: list of processed image tensors
                - image_sizes: tensor of (H, W) per image
                - state, embodiment_id, etc.
        """
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        
        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            
            if key == "vlm_content":
                # Handle vlm_content: extract text for tokenization, images for processing
                text_list = []
                all_images = []
                
                for v in values:
                    text_list.append(v["text"])
                    
                    # Extract images from conversation
                    conversation = v.get("conversation", [])
                    images = self.extract_images_from_conversation(conversation)
                    all_images.extend(images)
                
                # Process images with PyTorch
                if all_images:
                    processed_images, image_sizes = self.process_images_torch(all_images)
                    # Add batch dimension: (C, H, W) -> (1, C, H, W) to match Eagle format
                    batch["pixel_values"] = [img.unsqueeze(0) for img in processed_images]
                    batch["image_sizes"] = torch.stack(image_sizes)
                
                # Get tokenization from VLM processor
                # We need to call the processor with dummy images to get input_ids/attention_mask
                # The tokenization depends on image count and text, not pixel values
                
                # Create dummy PIL images with same dimensions as our tensor images
                from PIL import Image
                dummy_images = []
                for img in all_images:
                    h, w = img.shape[0], img.shape[1]
                    # Create minimal dummy image
                    dummy = Image.new('RGB', (w, h), color=(0, 0, 0))
                    dummy_images.append(dummy)
                
                # Build conversation with dummy PIL images for tokenization
                dummy_conversations = []
                img_idx = 0
                for v in values:
                    conversation = v.get("conversation", [])
                    dummy_conv = []
                    for message in conversation:
                        if not isinstance(message, dict):
                            dummy_conv.append(message)
                            continue
                        content = message.get("content", [])
                        if not isinstance(content, list):
                            dummy_conv.append(message)
                            continue
                        new_content = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                # Replace tensor with dummy PIL
                                new_content.append({"type": "image", "image": dummy_images[img_idx]})
                                img_idx += 1
                            else:
                                new_content.append(item)
                        dummy_conv.append({**message, "content": new_content})
                    dummy_conversations.append(dummy_conv)
                
                # Use process_vision_info with dummy images
                if self.model_type == "eagle":
                    processed_dummy_images, _ = self.processor.process_vision_info(dummy_conversations)
                else:
                    processed_dummy_images = dummy_images
                
                # Get tokenization (input_ids, attention_mask)
                vlm_inputs = self.processor(
                    text=text_list,
                    images=processed_dummy_images,
                    return_tensors="pt",
                    padding=True,
                )
                
                # Copy tokenization outputs (but NOT pixel_values - we use our own)
                for k, v in vlm_inputs.items():
                    if k not in ("pixel_values", "image_sizes"):
                        batch[k] = v
                        
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                if isinstance(values[0], np.ndarray) or isinstance(values[0], int):
                # state, state_mask, action and action_mask - stack to form batch dimension
                    batch[key] = torch.from_numpy(np.stack(values))
                else:
                    batch[key] = torch.stack(values)
        
        return {"inputs": batch}
    
    def __str__(self):
        return f"Gr00tN1d6DataCollatorTorch(model_name={self.model_name}, model_type={self.model_type})"


def create_torch_collator(
    model_name: str,
    model_type: Literal["eagle"] = "eagle",
    transformers_loading_kwargs: dict = {},
) -> Gr00tN1d6DataCollatorTorch:
    """
    Factory function to create a PyTorch-traceable collator.
    
    Args:
        model_name: HuggingFace model name
        model_type: VLM backbone type
        transformers_loading_kwargs: Kwargs for transformers loading
        
    Returns:
        Gr00tN1d6DataCollatorTorch instance
    """
    return Gr00tN1d6DataCollatorTorch(
        model_name=model_name,
        model_type=model_type,
        transformers_loading_kwargs=transformers_loading_kwargs,
    )
