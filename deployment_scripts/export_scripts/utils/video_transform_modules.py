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
