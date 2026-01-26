import torch
import torch.nn as nn
from typing import List, Dict


class ConcatTransform(nn.Module):
    """Exportable version of ConcatTransform."""

    def __init__(self,
                 video_concat_order: List[str] = None,
                 state_concat_order: List[str] = None,
                 action_concat_order: List[str] = None,
                 state_dims: Dict[str, int] = None,
                 action_dims: Dict[str, int] = None,
                 backward: bool = False,
                 **kwargs):
        """Initialize the ConcatTransform module.

        Args:
            video_concat_order: Concatenation order for each video modality
            state_concat_order: Concatenation order for each state modality
            action_concat_order: Concatenation order for each action modality
            state_dims: The dimensions of the state keys
            action_dims: The dimensions of the action keys
            backward: Whether to use the backward method as the forward method
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.video_concat_order = video_concat_order

        # Initialize with empty lists if None to avoid None type issues
        self.state_concat_order = state_concat_order if state_concat_order is not None else []
        self.action_concat_order = action_concat_order if action_concat_order is not None else []

        # Initialize with empty dicts if None
        self.state_dims = state_dims if state_dims is not None else {}
        self.action_dims = action_dims if action_dims is not None else {}

        if backward:
            self.forward = self.backward

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply concatenation to the specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with concatenated tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Process video modality
        if len(self.video_concat_order) > 0:
            video_keys_present = True
            for video_key in self.video_concat_order:
                if video_key not in result:
                    video_keys_present = False
                    break

            if video_keys_present:
                unsqueezed_videos = []

                for video_key in self.video_concat_order:
                    video_data = result.pop(video_key)
                    # [..., H, W, C] -> [..., 1, H, W, C]
                    if video_data.shape[0] != 1 or video_data.shape[1] != 1:
                        raise ValueError(
                            f"Video data for {video_key} has unexpected shape {video_data.shape}\n"
                            f"this module only supports T=1 V=1")

                    unsqueezed_video = video_data.squeeze().unsqueeze(-4)
                    unsqueezed_videos.append(unsqueezed_video)

                # Concatenate along the new axis
                result["video"] = torch.cat(
                    unsqueezed_videos, dim=-4)  # [..., V, H, W, C]
                result["video"] = result["video"].to(torch.float16)

        # Process state modality
        if len(self.state_concat_order) > 0:
            state_keys_present = True
            for state_key in self.state_concat_order:
                if state_key not in result:
                    state_keys_present = False
                    break

            if state_keys_present:
                state_tensors = []

                for state_key in self.state_concat_order:
                    state_tensors.append(result.pop(state_key))

                # Concatenate the state keys along the last dimension
                result["state"] = torch.cat(state_tensors, dim=-1)
                result["state"] = result["state"].to(torch.float16)

        # Process action modality
        if len(self.action_concat_order) > 0:
            action_keys_present = True
            for action_key in self.action_concat_order:
                if action_key not in result:
                    action_keys_present = False
                    break

            if action_keys_present:
                action_tensors = []

                for action_key in self.action_concat_order:
                    action_tensors.append(result.pop(action_key))

                # Concatenate the action keys along the last dimension
                result["action"] = torch.cat(action_tensors, dim=-1)
                result["action"] = result["action"].to(torch.float16)

        return result

    def backward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Process action tensor if present
        if "action" in data:
            start_dim = 0
            action_tensor = data.pop("action")

            # Make sure we have the configuration needed for unapplying
            assert (hasattr(self, "action_concat_order") and
                    self.action_concat_order is not None), "action_concat_order is required"
            assert hasattr(
                self, "action_dims") and self.action_dims is not None, "action_dims is required"

            # Split action tensor according to action_dims
            for key in self.action_concat_order:
                if key not in self.action_dims:
                    raise ValueError(
                        f"Action dim {key} not found in action_dims.")
                end_dim = start_dim + self.action_dims[key]
                data[key] = action_tensor[..., start_dim:end_dim]
                start_dim = end_dim

        # Process state tensor if present
        if ("state" in data and hasattr(self, "state_concat_order") and
                self.state_concat_order is not None):
            start_dim = 0
            state_tensor = data.pop("state")

            # Make sure we have the configuration needed for unapplying state
            assert hasattr(
                self, "state_dims") and self.state_dims is not None, "state_dims is required"

            # Split state tensor according to state_dims
            for key in self.state_concat_order:
                if key not in self.state_dims:
                    raise ValueError(
                        f"State dim {key} not found in state_dims.")
                end_dim = start_dim + self.state_dims[key]
                data[key] = state_tensor[..., start_dim:end_dim]
                start_dim = end_dim

        return data

    def inverse_transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split concatenated tensors back into original components.

        Args:
            data: Dictionary with concatenated tensors

        Returns:
            Dictionary with tensors split back into original components
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Split action tensor if present
        if "action" in result and len(self.action_concat_order) > 0:
            action_tensor = result.pop("action")
            start_dim = 0

            for action_key in self.action_concat_order:
                if action_key in self.action_dims:
                    end_dim = start_dim + self.action_dims[action_key]
                    result[action_key] = action_tensor[..., start_dim:end_dim]
                    start_dim = end_dim

        # Split state tensor if present
        if "state" in result and len(self.state_concat_order) > 0:
            state_tensor = result.pop("state")
            start_dim = 0

            for state_key in self.state_concat_order:
                if state_key in self.state_dims:
                    end_dim = start_dim + self.state_dims[state_key]
                    result[state_key] = state_tensor[..., start_dim:end_dim]
                    start_dim = end_dim

        return result
