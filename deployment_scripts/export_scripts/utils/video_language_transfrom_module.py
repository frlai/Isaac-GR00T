
import torch
import torch.nn as nn

from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode

from typing import List, Dict, Tuple


class Eagle2VideoTransform(nn.Module):
    """Exportable version of Eagle2Process."""

    int_to_interpolation_string = {
        0: InterpolationMode.NEAREST,    # PIL.Image.Resampling.NEAREST
        1: InterpolationMode.LANCZOS,    # PIL.Image.Resampling.LANCZOS
        2: InterpolationMode.BILINEAR,   # PIL.Image.Resampling.BILINEAR
        3: InterpolationMode.BICUBIC,    # PIL.Image.Resampling.BICUBIC
        4: InterpolationMode.BOX,        # PIL.Image.Resampling.BOX
        5: InterpolationMode.HAMMING,    # PIL.Image.Resampling.HAMMING
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
        self.do_resize = configs['do_resize'] if configs['do_resize'] else False
        self.do_center_crop = configs['do_center_crop'] if configs['do_center_crop'] else False
        self.do_rescale = configs['do_rescale'] if configs['do_rescale'] else False
        self.do_normalize = configs['do_normalize'] if configs['do_normalize'] else False
        self.rescale_factor = configs['rescale_factor']
        self.image_mean = configs['image_mean']
        self.image_std = configs['image_std']
        self.do_pad = configs['do_pad']
        self.min_dynamic_tiles = configs['min_dynamic_tiles']
        self.max_dynamic_tiles = configs['max_dynamic_tiles']

    def process_frames(self, data: torch.Tensor) -> List[torch.Tensor]:
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

    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        orig_width: int,
        orig_height: int,
        tile_size: int
    ) -> Tuple[int, int]:
        """Find the closest aspect ratio considering both ratio and area."""
        best_factor = -1.0
        best_ratio = (1, 1)
        area = orig_width * orig_height

        for ratio in target_ratios:
            target_aspect_ratio = float(ratio[0]) / float(ratio[1])

            # Calculate factor based on area and ratio
            factor_based_on_area_n_ratio = min(
                float(ratio[0] * ratio[1] * tile_size *
                      tile_size) / float(area), 0.6
            ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)

            if factor_based_on_area_n_ratio > best_factor:
                best_factor = factor_based_on_area_n_ratio
                best_ratio = ratio

        return best_ratio

    def _get_patch_output_size(self, target_resolution: Tuple[int, int],
                               original_height: int, original_width: int) -> Tuple[int, int]:
        """
        Calculate output size when resizing image to target resolution while preserving aspect ratio.
        Assumes channel-first format: [C, H, W] or [B, C, H, W].

        Args:
            image: Input image tensor in channel-first format
            target_resolution: Tuple of (target_height, target_width)
            original_height: Optional original height, if not provided will extract from image
            original_width: Optional original width, if not provided will extract from image

        Returns:
            Tuple of (new_height, new_width) - the calculated output dimensions
        """

        target_height, target_width = target_resolution

        scale_w = float(target_width) / float(original_width)
        scale_h = float(target_height) / float(original_height)

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(
                int(torch.ceil(torch.tensor(original_height * scale_w)).item()), target_height)
        else:
            new_height = target_height
            new_width = min(
                int(torch.ceil(torch.tensor(original_width * scale_h)).item()), target_width)

        return new_height, new_width

    def _crop(self, img: torch.Tensor, left: int, top: int, right: int, bottom: int) -> torch.Tensor:
        img_height = img.shape[1]
        img_width = img.shape[2]
        if top < 0 or left < 0 or bottom > img_height or right > img_width:
            raise ValueError("Crop coordinates out of bounds")

        if top >= bottom or left >= right:
            raise ValueError("Invalid crop coordinates")

        return img[:, top:bottom, left:right]

    def _get_image_patches(self, image: torch.Tensor) -> List[torch.Tensor]:
        min_num = self.min_dynamic_tiles
        max_num = self.max_dynamic_tiles
        tile_size = self.tile_size
        use_thumbnail = self.use_thumbnail
        interpolation = self.interpolation
        pad_during_tiling = self.pad_during_tiling

        orig_height = int(image.shape[-2])
        orig_width = int(image.shape[-1])
        aspect_ratio = float(orig_width) / float(orig_height)

        # calculate the existing image aspect ratio
        # Calculate target ratios efficiently (avoiding duplicates)
        target_ratios: List[Tuple[int, int]] = []
        for i in range(1, max_num + 1):
            for j in range(1, max_num + 1):
                total_tiles = i * j
                if min_num <= total_tiles <= max_num:
                    target_ratios.append((i, j))

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )
        # calculate the target width and height
        target_width = int(tile_size * target_aspect_ratio[0])
        target_height = int(tile_size * target_aspect_ratio[1])
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        if pad_during_tiling:
            new_height, new_width = self._get_patch_output_size(
                (target_height, target_width), orig_height, orig_width)
            resized_image = F.resize(
                image, [new_height, new_width], interpolation=interpolation
            )
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded_image = F.pad(resized_image, padding=[
                                 paste_x, paste_y, paste_x, paste_y])
            image_used_to_split = padded_image
        else:
            image_used_to_split = F.resize(
                image, [target_height, target_width], interpolation=interpolation
            )
        processed_tiles = []
        for i in range(blocks):
            box = (
                (i % (target_width // tile_size)) * tile_size,
                (i // (target_width // tile_size)) * tile_size,
                ((i % (target_width // tile_size)) + 1) * tile_size,
                ((i // (target_width // tile_size)) + 1) * tile_size,
            )
            # split the image
            split_img = self._crop(image_used_to_split,
                                   box[0], box[1], box[2], box[3])
            processed_tiles.append(split_img)

        if use_thumbnail and len(processed_tiles) != 1:
            thumbnail_img = F.resize(
                image, (tile_size, tile_size), interpolation=interpolation)
            processed_tiles.append(thumbnail_img)
        return processed_tiles

    def _pad_for_batching(
        self,
        pixel_values: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(
                image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """In eval mode, just return the input.

        Args:
            data: Dictionary of video tensors
            **kwargs: Additional keyword arguments
        """
        if "video" not in data:
            return {}

        frames = self.process_frames(data['video'])
        image_sizes = []
        processed_frames = []

        for frame in frames:
            if frame.ndim != 3:
                raise ValueError(
                    "Frame must be 3D, got " + str(frame.ndim) + "D")

            image_patches = self._get_image_patches(frame)

            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = self.group_images_by_shape(
                image_patches
            )

            for shape, stacked_image_patches in grouped_image_patches.items():
                if self.do_resize:
                    target_size = (self.size['height'], self.size['width'])
                    stacked_image_patches = F.resize(
                        stacked_image_patches, target_size, interpolation=self.interpolation
                    )
                if self.do_center_crop:
                    stacked_image_patches = F.center_crop(
                        stacked_image_patches, (self.tile_size, self.tile_size)
                    )

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

            processed_image_patches_stacked = torch.stack(
                processed_image_patches, dim=0)

            processed_frames.append(processed_image_patches_stacked)
            image_sizes.append(torch.tensor([[frame.shape[1], frame.shape[2]]]))

        if self.do_pad:
            processed_frames = self._pad_for_batching(processed_frames)

        processed_frames_stacked = torch.cat(processed_frames, dim=0)
        image_sizes_stacked = torch.cat(image_sizes, dim=0)

        image_sizes_stacked = image_sizes_stacked.to(
            processed_frames_stacked.device)
        return {"eagle_pixel_values": processed_frames_stacked, "eagle_image_sizes": image_sizes_stacked}

    def group_images_by_shape(self, images: List[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[int, Tuple[str, int]]]:
        """
        Groups images by their shapes (height, width) for efficient batch processing.
        TorchScript compatible implementation.

        This method organizes a list of images into groups based on their dimensions,
        facilitating efficient batch processing as images of the same size can be 
        processed together without resizing or padding.

        Args:
            images (List[torch.Tensor]): List of image tensors to be grouped.
                Each tensor should be in format [C, H, W].

        Returns:
            Tuple containing:
                - Dict[str, torch.Tensor]: Dictionary where keys are "HxW" strings 
                  and values are stacked tensors of images with that shape.
                - Dict[int, Tuple[str, int]]: Dictionary mapping original image 
                  indices to their position in the grouped structure. Each value is a tuple 
                  containing (shape_key, index_within_shape_group).
        """
        # Group images by shape using regular dict (TorchScript compatible)
        grouped_images: Dict[str, List[torch.Tensor]] = {}
        indices: Dict[int, Tuple[str, int]] = {}

        for idx, image in enumerate(images):
            # Extract (height, width) from tensor shape [C, H, W]
            if image.ndim >= 2:
                height = int(image.shape[-2])
                width = int(image.shape[-1])
                shape_key = str(height) + "x" + str(width)
            else:
                raise ValueError("Image at index " +
                                 str(idx) + " has invalid shape")

            # Add image to the appropriate group
            if shape_key not in grouped_images:
                grouped_images[shape_key] = []
            grouped_images[shape_key].append(image)

            # Track the index within this shape group
            shape_group_index = len(grouped_images[shape_key]) - 1
            indices[idx] = (shape_key, shape_group_index)

        # Stack images in each group for efficient processing
        stacked_grouped_images: Dict[str, torch.Tensor] = {}
        for shape_key in grouped_images:
            image_list = grouped_images[shape_key]
            # Stack all images of the same shape into a single tensor
            stacked_grouped_images[shape_key] = torch.stack(image_list, dim=0)

        return stacked_grouped_images, indices

    def reorder_images(self, grouped_images: Dict[str, torch.Tensor],
                       indices: Dict[int, Tuple[str, int]]) -> List[torch.Tensor]:
        """
        Reorders processed images back to their original sequence.
        TorchScript compatible implementation.

        This method reconstructs the original image order after batch processing
        images that were grouped by shape.

        Args:
            grouped_images (Dict[str, torch.Tensor]): Dictionary of 
                processed image groups where keys are "HxW" strings and 
                values are stacked tensors.
            indices (Dict[int, Tuple[str, int]]): Dictionary mapping 
                original image indices to their position in the grouped structure.

        Returns:
            List[torch.Tensor]: List of images in their original order.
        """
        reordered_images: List[torch.Tensor] = []

        # Get the number of images to reorder
        num_images = len(indices)

        # Process images in original order (0, 1, 2, ...)
        for idx in range(num_images):
            shape_key, shape_group_index = indices[idx]
            # Extract the specific image from the stacked tensor
            stacked_images = grouped_images[shape_key]
            image = stacked_images[shape_group_index]
            reordered_images.append(image)

        return reordered_images

    def rescale_and_normalize(self, image: torch.Tensor, do_rescale: bool,
                              rescale_factor: float, do_normalize: bool,
                              image_mean: List[float], image_std: List[float]) -> torch.Tensor:
        """
        Rescale and normalize image.
        TorchScript compatible implementation.

        Args:
            image: Input image tensor
            do_rescale: Whether to rescale
            rescale_factor: Factor to rescale by
            do_normalize: Whether to normalize
            image_mean: Mean values for normalization
            image_std: Standard deviation values for normalization

        Returns:
            Rescaled and normalized image tensor
        """
        # Convert to float32 first (like BaseImageProcessorFast does)
        image = image.to(torch.float32)

        if do_rescale:
            image = image * rescale_factor

        if do_normalize:
            # Convert lists to tensors for normalization
            mean = torch.tensor(
                image_mean, dtype=image.dtype, device=image.device)
            std = torch.tensor(image_std, dtype=image.dtype,
                               device=image.device)

            # Reshape for broadcasting: [C, 1, 1] for [N, C, H, W] tensors
            if image.ndim == 4:  # [N, C, H, W]
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
            elif image.ndim == 3:  # [C, H, W]
                mean = mean.view(-1, 1, 1)
                std = std.view(-1, 1, 1)

            image = (image - mean) / std

        return image


class Gr00tVideoLanguageTransform(nn.Module):
    """Exportable version of Gr00tVideoLanguage."""

    def __init__(self, **kwargs):
        super().__init__()

        # known outputs of the gr00t transform module that will be used:
        self.known_outputs = [
            'eagle_input_ids', 'eagle_image_sizes', 'eagle_attention_mask', 'embodiment_id']
        self.video_transform = Eagle2VideoTransform(**kwargs)
        self.gr00t_transform_outputs = {}

    def set_gr00t_transform_outputs(self, expected_outputs: Dict[str, torch.Tensor]):
        for key in self.known_outputs:
            if key not in expected_outputs:
                raise ValueError(f"Expected output {key} not found")
            self.gr00t_transform_outputs[key] = expected_outputs[key].to(
                torch.int32)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        video_transform_outputs = self.video_transform(data)
        if "eagle_image_sizes" in video_transform_outputs:
            video_transform_outputs["eagle_image_sizes"] = video_transform_outputs["eagle_image_sizes"].to(
                torch.int32)
        device = video_transform_outputs["eagle_pixel_values"].device

        outputs = {}
        for key in self.gr00t_transform_outputs:
            if self.gr00t_transform_outputs[key].device != device:
                outputs[key] = self.gr00t_transform_outputs[key].to(device)
            else:
                outputs[key] = self.gr00t_transform_outputs[key]

        for key in video_transform_outputs:
            outputs[key] = video_transform_outputs[key]

        return outputs
