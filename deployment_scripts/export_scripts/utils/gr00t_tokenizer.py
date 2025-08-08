import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from .eagle2_hg_model.inference_eagle_repo import EagleProcessor

# Default system message used in prompts
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

EAGLE_KEYS = ["pixel_values", "input_ids", "attention_mask"]


def collate_gr00t(features: List[dict], processor, device: str) -> dict:
    batch = {}
    keys = features[0].keys()
    assert all(
        all(key in feature for key in keys) for feature in features
    ), "All features must have the same keys."

    assert all(key in EAGLE_KEYS for key in keys), f"unexpected keys: {keys=}"

    vlm_batch = processor.collate_fn(features)

    # Create the batch dictionary from the VLM processor output
    batch = {}
    # Convert all values in vlm_batch to the appropriate device if they are tensors
    for key, value in vlm_batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        else:
            batch[key] = value

    return batch


class GR00TTransform(nn.Module):
    """
    Handles video and language processing for GR00T models.
    This complements the TorchScript-compatible GR00TTransform which only handles state processing.
    Implemented as a PyTorch nn.Module to fit into model pipelines.
    """

    def __init__(
        self,
        vlm_processor_metadata: Dict = {},
        default_instruction: str = "Perform the default behavior.",
        language_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the GR00TTokenizer.

        Args:
            vlm_processor: The processor for vision-language processing
            (must implement prepare_input method)
            default_instruction: Default instruction to use if language is not provided
            language_key: The key for language in the input data
            **kwargs: Additional arguments for compatibility
        """
        super(GR00TTransform, self).__init__()
        self.vlm_processor = EagleProcessor(**vlm_processor_metadata)
        self.default_instruction = default_instruction
        self.language_key = language_key

    def check_language_key(self, data: Dict) -> None:
        """
        Extract language key if present in the data.

        Args:
            data: Input data dictionary
        """
        # Group keys by modality
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except Exception:  # noqa: E722
                # Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        # Handle language key
        if "language" in grouped_keys and len(grouped_keys["language"]) > 0:
            language_keys = grouped_keys["language"]
            if len(language_keys) == 1:
                self.language_key = language_keys[0]

    def _prepare_video(self, data: Dict) -> np.ndarray:
        """
        Process, stack, and pad images from data['video'].

        Args:
            data: Input data dictionary

        Returns:
            Processed video frames
        """
        video = data["video"]  # [t v h w c]
        return video

    def _prepare_language(self, data: Dict) -> str:
        """
        Extract and potentially transform language input.

        Args:
            data: Input data dictionary

        Returns:
            Processed language instruction
        """
        if self.language_key is not None and self.language_key in data:
            raw_language = data[self.language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]
        else:
            raw_language = self.default_instruction

        return raw_language

    def _apply_gr00t_processing(self, images: np.ndarray, language: str) -> Dict:
        """
        Apply VLM processing to images and language.

        Args:
            images: Processed video frames
            language: Processed language instruction

        Returns:
            Dictionary with processed inputs for the VLM model
        """
        # Ensure images have the right shape
        if not images.shape[0] == 1:
            raise ValueError(
                "Expected single timestep, check formatting when doing multi-time step")
        # Remove the singleton time dimension
        images = images[0]
        # Convert tensor to numpy array if needed
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        images = [{"np_array": images[idx]} for idx in range(len(images))]
        # Create prompt with system message and user content
        prompt = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": language,
                "image": images,
            },
        ]

        # Process with VLM processor
        inputs = self.vlm_processor.prepare_input({"prompt": prompt})
        return inputs

    def process(self, data: Dict, device: str) -> Dict:
        """
        Process video and language inputs.

        Args:
            data: Input data dictionary (non-batched)

        Returns:
            Dictionary with processed inputs
        """
        # Check for language key in data
        self.check_language_key(data)
        # 1) Extract and process video and language
        images = self._prepare_video(data)

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if isinstance(images, np.ndarray):
            images = images.astype(np.uint8)

        language = self._prepare_language(data)

        # 2) Apply VLM processing
        vlm_outputs = self._apply_gr00t_processing(images, language)
        # 3) Return processed outputs
        result = {}
        for k, v in vlm_outputs.items():
            result[k] = v.to(device=device)

        return result

    def process_batch(self, data: Dict, batch_size: int, device: str) -> Dict:
        """
        Process a batch of data.

        Args:
            data: Input data dictionary
            batch_size: Batch size

        Returns:
            Dictionary with processed inputs
        """
        # Split on batch dimension.
        data_split = []
        for i in range(batch_size):
            single_data = {}
            for key, value in data.items():
                # Special handling for string values to prevent character-wise splitting
                if isinstance(value, str):
                    # For string values, keep the entire string instead of indexing
                    single_data[key] = value
                else:
                    # For arrays and other data types, extract the i-th element
                    try:
                        single_data[key] = value[i]
                    except (TypeError, IndexError):
                        # If value is not indexable or index is out of bounds, use the whole value
                        single_data[key] = value
            data_split.append(single_data)

        # Process each element.
        data_split_processed = [self.process(
            elem, device) for elem in data_split]

        return collate_gr00t(data_split_processed, self.vlm_processor, device)

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except Exception:  # noqa: E722
                # Handle language annotation special case
                if "annotation" in key:
                    modality = "language"
                else:
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(
                f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def forward(self, data: Dict) -> Dict:
        """
        Forward method for nn.Module compatibility.
        Process the input data.

        Args:
            data: Input data dictionary

        Returns:
            Dictionary with processed inputs
        """

        device = 'cpu'
        key_devices = []
        # If the number of dimensions on "image" is less than 6, unsqueeze until there are 6 dimensions
        if "video" in data and data["video"].ndim == 4:
            while data["video"].ndim < 6:
                data["video"] = data["video"].unsqueeze(0)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape, value.dtype)
            else:
                print(key, value)
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                key_devices.append(data[key].device)

        if all(device == key_devices[0] for device in key_devices):
            device = key_devices[0]

        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            retval = self.process_batch(data, batch_size, device)
        else:
            retval = self.process(data, device)

        for k, v in retval.items():
            if v.dtype == torch.bfloat16:
                retval[k] = v.to(torch.float16)
            elif v.dtype == torch.int64:
                retval[k] = v.to(torch.int32)

        return retval
