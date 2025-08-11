import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from einops import rearrange
from typing import List, Dict


class VideoBase(nn.Module):
    """Base class for video transform modules."""

    def __init__(self, apply_to: List[str], **kwargs):
        """Initialize with list of keys to apply transform to.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Base forward method.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with transformed tensors
        """
        return data


class VideoCrop(nn.Module):
    """Exportable version of VideoCrop."""

    def __init__(self, height: int, width: int, scale: float, apply_to: List[str], **kwargs):
        """Initialize the VideoCrop module.

        Args:
            height: The height of the input video
            width: The width of the input video
            scale: The scale of the crop
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.height = height
        self.width = width
        self.scale = scale
        self.apply_to = apply_to

        # Compute crop size
        crop_height = int(self.height * self.scale)
        crop_width = int(self.width * self.scale)
        self.size = (crop_height, crop_width)

        # Create transform for eval mode (center crop)
        self.transform = torch.jit.script(T.CenterCrop(self.size))

    def transform_tensor(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply center crop to a single video tensor.

        Args:
            x: Video tensor of shape [T, C, H, W],
            [B, C, H, W], [T, B, C, H, W], or other dimensions
            **kwargs: Additional keyword arguments

        Returns:
            Cropped video tensor with same dimensions
        """
        # Store original shape and dimensionality
        orig_shape = x.shape
        orig_ndim = len(orig_shape)

        # Handle different input dimensions
        if orig_ndim == 4:
            # Standard case: [T, C, H, W] or [B, C, H, W]
            # Apply transform directly
            x = self.transform(x)
        elif orig_ndim == 5:
            # [T, B, C, H, W] case
            b = orig_shape[1]
            # Reshape to [(T*B), C, H, W]
            x = rearrange(x, 't b c h w -> (t b) c h w')
            # Apply transform
            x = self.transform(x)
            # Reshape back
            x = rearrange(x, '(t b) c h w -> t b c h w', b=b)
        elif orig_ndim == 3:
            # Could be [C, H, W] or [T, H, W] or other 3D format
            # If it's [C, H, W], apply transform directly
            if orig_shape[0] <= 3:  # Assume it's channels if <= 3
                x = self.transform(x)
            else:
                # Assume [T, H, W] format (without channels), add channel dim
                x = x.unsqueeze(1)  # [T, 1, H, W]
                x = self.transform(x)
                x = x.squeeze(1)  # [T, H', W']
        elif orig_ndim == 2:
            # Could be [H, W], add dimensions for transform
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            x = self.transform(x)
            x = x.squeeze(0).squeeze(0)  # [H', W']
        else:
            # For any other dimension case, try to identify the H,W dimensions
            # and reshape accordingly
            raise ValueError(
                f"Unsupported input dimension: {orig_ndim}. Shape: {orig_shape}")

        return x

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply center crop to all specified video tensors in the dictionary.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with transformed video tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply transform to each key in apply_to
        for key in self.apply_to:
            if key in result:
                result[key] = self.transform_tensor(result[key], **kwargs)

        return result


class VideoResize(nn.Module):
    """Exportable version of VideoResize."""

    def __init__(self, height: int, width: int,
                 apply_to: List[str], interpolation: str = "linear",
                 antialias: bool = True, **kwargs):
        """Initialize the VideoResize module.

        Args:
            height: Target height
            width: Target width
            apply_to: List of keys in the input dictionary to apply the transform to
            interpolation: Interpolation mode
            antialias: Whether to use antialiasing
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.height = height
        self.width = width
        self.apply_to = apply_to

        # Convert interpolation string to torchvision mode
        interp_map = {
            "nearest": T.InterpolationMode.NEAREST,
            "linear": T.InterpolationMode.BILINEAR,
            "cubic": T.InterpolationMode.BICUBIC,
            "lanczos4": T.InterpolationMode.LANCZOS,
            "nearest_exact": T.InterpolationMode.NEAREST_EXACT
        }

        interpolation_mode = interp_map.get(
            interpolation, T.InterpolationMode.BILINEAR)

        self.interpolation = interpolation
        self.interpolation_mode = interpolation_mode
        self.antialias = antialias

        self.transform = torch.jit.script(T.Resize(
            (self.height, self.width),
            interpolation=interpolation_mode,
            antialias=antialias
        ))

    def transform_tensor(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Resize a single video tensor.

        Args:
            x: Video tensor of shape [T, C, H, W], [B, C, H, W],
            [T, B, C, H, W], or other dimensions
            **kwargs: Additional keyword arguments

        Returns:
            Resized video tensor with same batch dimensions
        """
        # Store original shape and dimensionality
        orig_shape = x.shape
        orig_ndim = len(orig_shape)

        # Handle different input dimensions
        if orig_ndim == 4:
            # Standard case: [T, C, H, W] or [B, C, H, W]
            # Apply transform directly
            x = self.transform(x)
        elif orig_ndim == 5:
            # [T, B, C, H, W] case
            b = orig_shape[1]
            # Reshape to [(T*B), C, H, W]
            x = rearrange(x, 't b c h w -> (t b) c h w')
            # Apply transform
            x = self.transform(x)
            # Reshape back
            x = rearrange(x, '(t b) c h w -> t b c h w', b=b)
        elif orig_ndim == 3:
            # Could be [C, H, W] or [T, H, W] or other 3D format
            # If it's [C, H, W], apply transform directly
            if orig_shape[0] <= 3:  # Assume it's channels if <= 3
                x = self.transform(x)
            else:
                # Assume [T, H, W] format (without channels), add channel dim
                x = x.unsqueeze(1)  # [T, 1, H, W]
                x = self.transform(x)
                x = x.squeeze(1)  # [T, H', W']
        elif orig_ndim == 2:
            # Could be [H, W], add dimensions for transform
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            x = self.transform(x)
            x = x.squeeze(0).squeeze(0)  # [H', W']
        else:
            # For any other dimension case, try to identify the H,W dimensions
            # and reshape accordingly
            raise ValueError(
                f"Unsupported input dimension: {orig_ndim}. Shape: {orig_shape}")

        return x

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply resize to all specified video tensors in the dictionary.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with transformed video tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply transform to each key in apply_to
        for key in self.apply_to:
            if key in result:
                result[key] = self.transform_tensor(result[key], **kwargs)

        return result


class VideoRandomRotation(nn.Module):
    """Exportable version of VideoRandomRotation."""

    def __init__(self, degrees, apply_to: List[str], interpolation: str = "linear", **kwargs):
        """Initialize the VideoRandomRotation module.

        Args:
            degrees: Rotation degrees range
            apply_to: List of keys in the input dictionary to apply the transform to
            interpolation: Interpolation mode for rotation
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoHorizontalFlip(nn.Module):
    """Exportable version of VideoHorizontalFlip."""

    def __init__(self, p: float, apply_to: List[str], **kwargs):
        """Initialize the VideoHorizontalFlip module.

        Args:
            p: Probability of horizontal flip
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoGrayscale(nn.Module):
    """Exportable version of VideoGrayscale."""

    def __init__(self, p: float, apply_to: List[str], **kwargs):
        """Initialize the VideoGrayscale module.

        Args:
            p: Probability of grayscale transform
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoColorJitter(nn.Module):
    """Exportable version of VideoColorJitter."""

    def __init__(self, brightness, contrast, saturation, hue, apply_to: List[str], **kwargs):
        """Initialize the VideoColorJitter module.

        Args:
            brightness: Brightness adjustment range
            contrast: Contrast adjustment range
            saturation: Saturation adjustment range
            hue: Hue adjustment range
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoRandomGrayscale(nn.Module):
    """Exportable version of VideoRandomGrayscale."""

    def __init__(self, p: float, apply_to: List[str], **kwargs):
        """Initialize the VideoRandomGrayscale module.

        Args:
            p: Probability of grayscale transform
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoRandomPosterize(nn.Module):
    """Exportable version of VideoRandomPosterize."""

    def __init__(self, bits: int, p: float, apply_to: List[str], **kwargs):
        """Initialize the VideoRandomPosterize module.

        Args:
            bits: Number of bits to keep
            p: Probability of posterize transform
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        # For eval mode, this transform does nothing

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments

        Returns:
            Input dictionary unchanged
        """
        return data


class VideoToTensor(nn.Module):
    """Exportable version of VideoToTensor."""

    def __init__(self, apply_to: List[str], **kwargs):
        """Initialize the VideoToTensor module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to

    def transform_tensor(self, frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert tensor format and type.

        Args:
            frames: Video tensor in [T, H, W, C] format
            **kwargs: Additional keyword arguments

        Returns:
            Converted tensor in [T, C, H, W] format, float32 type with values in [0,1] range
        """
        # Ensure the tensor is float32
        if frames.dtype != torch.float32:
            frames = frames.to(torch.float32)

        # If the values are in [0,255] range, scale them to [0,1]
        max_val = frames.max()
        if max_val > 1.0:
            frames = frames / 255.0

        # Handle dimension permutation based on input shape
        if frames.ndim == 4:
            frames = frames.permute(0, 3, 1, 2)
        elif frames.ndim == 5:
            frames = frames.permute(0, 1, 4, 2, 3)

        return frames

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Convert specified tensors in the dictionary to float32 and permute dimensions.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with converted tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply transform to each key in apply_to
        for key in self.apply_to:
            result[key] = self.transform_tensor(result[key], **kwargs)

        return result


class VideoToNumpy(nn.Module):
    """Exportable version of VideoToNumpy.

    Since numpy is not supported with torch.export, this module is adapted
    to keep everything as torch tensors.
    """

    def __init__(self, apply_to: List[str], **kwargs):
        """Initialize the VideoToNumpy module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to

    def transform_tensor(self, frames: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert tensor format without converting to numpy.

        Args:
            frames: Video tensor in [T, C, H, W] format
            **kwargs: Additional keyword arguments

        Returns:
            Tensor in [T, H, W, C] format, with values in uint8 format (0-255 range)
        """
        # Convert from [T, C, H, W] to [T, H, W, C] and scale to 0-255 range
        # Handle dimension permutation based on input shape
        if frames.ndim == 4:
            frames = frames.permute(0, 2, 3, 1)
        elif frames.ndim == 5:
            frames = frames.permute(0, 1, 3, 4, 2)

        frames = frames * 255.0

        # Convert to uint8 type
        frames = frames.to(torch.uint8)

        return frames

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply format conversion to specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with converted tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply transform to each key in apply_to
        for key in self.apply_to:
            result[key] = self.transform_tensor(result[key], **kwargs)

        return result


"""
- image_processor: Eagle2_5_VLImageProcessorFast {
  "auto_map": {
    "AutoImageProcessor": "image_processing_eagle2_5_vl_fast.Eagle2_5_VLImageProcessorFast",
    "AutoProcessor": "processing_eagle2_5_vl.Eagle2_5_VLProcessor"
  },
  "crop_size": null,
  "data_format": "channels_first",
  "default_to_square": false,
  "device": null,
  "do_center_crop": null,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": false,
  "do_rescale": true,
  "do_resize": false,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "Eagle2_5_VLImageProcessorFast",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "input_data_format": null,
  "max_dynamic_tiles": 12,
  "min_dynamic_tiles": 1,
  "pad_during_tiling": false,
  "processor_class": "Eagle2_5_VLProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "return_tensors": null,
  "size": {
    "height": 224,
    "width": 224
  },
  "tokens_per_tile": 256,
  "use_thumbnail": true
}"""


class GR00TTransform(nn.Module):
    """Exportable version of Eagle2Process."""

    int_to_interpolation_string = {
        0: 'nearest',    # PIL.Image.Resampling.NEAREST
        1: 'lanczos',    # PIL.Image.Resampling.LANCZOS
        2: 'bilinear',   # PIL.Image.Resampling.BILINEAR
        3: 'bicubic',    # PIL.Image.Resampling.BICUBIC
        4: 'box',        # PIL.Image.Resampling.BOX
        5: 'hamming',    # PIL.Image.Resampling.HAMMING
    }

    def __init__(self, eagle_processor, **kwargs):
        super().__init__()
        configs = eagle_processor.__dict__['image_processor'].__dict__
        self.interpolation = self.int_to_interpolation_string[configs['resample']]
        if configs['size'] and configs['size']['height'] and configs['size']['width']:
            self.size_tuple = (
                configs['size']['height'], configs['size']['width'])
        else:
            self.size_tuple = (configs['size']['shortest_edge'],
                               configs['size']['shortest_edge'])

        if configs['crop_size'] and configs['crop_size']['height']:
            self.tile_size = configs['crop_size']['height']
        else:
            self.tile_size = configs['size']['height']
        self.size = configs['size']
        self.use_thumbnail = configs['use_thumbnail']
        self.pad_during_tiling = configs['pad_during_tiling']
        self.do_resize = configs['do_resize']
        self.do_center_crop = configs['do_center_crop']
        self.do_rescale = configs['do_rescale']
        self.do_normalize = configs['do_normalize']
        self.rescale_factor = configs['rescale_factor']
        self.image_mean = configs['image_mean']
        self.image_std = configs['image_std']
        self.return_tensors = configs['return_tensors']
        self.do_pad = configs['do_pad']
        self.min_dynamic_tiles = configs['min_dynamic_tiles']
        self.max_dynamic_tiles = configs['max_dynamic_tiles']

    def process_frames(self, data: torch.Tensor) -> torch.Tensor:
        video = data

        # Step 1: Normalize video dims - if 4D, unsqueeze until 6D [B, T, V, H, W, C]
        # Add batch and view dimensions
        while video.ndim < 6:
            video = video.unsqueeze(0)

        batch_size = video.shape[0]
        if (batch_size != 1):
            raise ValueError(
                f"Batch size must be 1, got {batch_size}")

        single_video = video.squeeze(0)

        # Step 4: Flatten frames and convert CHW format for processing
        v, t, h, w, c = single_video.shape
        # convert to chw
        single_video = single_video.permute(0, 1, 4, 2, 3)
        # Flatten into frames: [V*T, C, H, W]
        frames_chw = single_video.reshape(v * t, c, h, w)

        processed_frames = list(torch.unbind(frames_chw, dim=0))

        return processed_frames

    def _get_image_patches(self, image: torch.Tensor) -> List[torch.Tensor]:
        min_num = self.min_dynamic_tiles
        max_num = self.max_dynamic_tiles
        size = self.size_tuple
        tile_size = self.tile_size
        use_thumbnail = self.use_thumbnail
        interpolation = self.interpolation
        pad_during_tiling = self.pad_during_tiling

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments
        """
        if not "video" in data:
            return data

        frames = self.process_frames(data['video'])
        processed_frames = []

        for frame in frames:
            image_patches = self._get_image_patches(frame)

            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = self.group_images_by_shape(
                image_patches
            )

            for shape, stacked_image_patches in grouped_image_patches.items():
                if self.do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=self.size,
                        interpolation=self.interpolation,
                    )
                if self.do_center_crop:
                    stacked_image_patches = self.center_crop(
                        stacked_image_patches, self.tile_size)
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches,
                    self.do_rescale,
                    self.rescale_factor,
                    self.do_normalize,
                    self.image_mean,
                    self.image_std,
                )
                processed_image_patches_grouped[shape] = stacked_image_patches

            processed_image_patches = self.reorder_images(
                processed_image_patches_grouped, grouped_image_patches_index
            )
            if self.return_tensors:
                processed_image_patches = torch.stack(
                    processed_image_patches, dim=0)

            processed_frames.append(processed_image_patches)

        if self.do_pad:
            processed_frames = self._pad_for_batching(processed_frames)

        if self.return_tensors:
            processed_frames = torch.cat(processed_frames, dim=0)

        return processed_frames
