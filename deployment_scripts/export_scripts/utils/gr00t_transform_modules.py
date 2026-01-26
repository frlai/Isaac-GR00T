import torch
import torch.nn as nn
from typing import Dict, Tuple


class GR00TTransform(nn.Module):
    """
    Exportable version of GR00TTransform for TorchScript compatibility.

    This version focuses only on state processing in eval mode.
    Video and language processing is handled separately.
    """

    _EMBODIMENT_TAG_MAPPING = {
        "gr1": 24,
        "new_embodiment": 31,  # use the last projector for new embodiment,
    }

    def __init__(
        self,
        max_state_dim: int,
        max_action_dim: int,
        state_horizon: int,
        action_horizon: int,
        embodiment_tag: int = 24,  # Default to gr1
        backward: bool = False,
        **kwargs
    ):
        """
        Initialize the GR00TTransform module.

        Args:
            max_state_dim: Maximum state dimension
            max_action_dim: Maximum action dimension
            (not used in forward, kept for API compatibility)
            state_horizon: Number of state timesteps
            action_horizon: Number of action timesteps
            (not used in forward, kept for API compatibility)
            embodiment_tag: Embodiment ID to use
            **kwargs: Additional keyword arguments for compatibility
        """
        super().__init__()
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.state_horizon = state_horizon
        self.action_horizon = action_horizon
        self.embodiment_id = torch.tensor(
            self._EMBODIMENT_TAG_MAPPING[embodiment_tag.value], dtype=torch.int64)

        # If backward is True, use the backward method as the forward method
        if backward:
            self.forward = self.backward

    def check_keys_and_batch_size(self, data):
        dims = []
        batch_sizes = []
        for key in data.keys():
            dim = data[key].ndim
            if dim == 3:
                dims.append(dim)
                batch_sizes.append(data[key].shape[0])
            elif dim == 2:
                dims.append(dim)
                batch_sizes.append(1)
            else:
                raise ValueError(f"Unsupported tensor dimensions: {dim}")

        # if len(set(dims)) > 1 or len(set(batch_sizes)) > 1:
        #     raise ValueError(
        #         f"Inconsistent tensor dims: {dims} batch sizes: {batch_sizes}")

        return dims[0] == 3, batch_sizes[0]

    def prepare_state(self,
                      data: Dict[str, torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare state tensors from input data.

        Args:
            data: Dictionary containing input tensors

        Returns:
            Tuple containing:
                - Padded state tensor
                - State mask tensor
                - Number of state tokens as a tensor
        """
        # Ensure state exists in the data - will fail if not present
        torch._assert("state" in data, "State must exist in input data")

        # Get state from data
        state = data["state"]

        # Ensure state has the correct time dimension
        torch._assert(state.shape[0] == self.state_horizon,
                      "State time dimension must match state_horizon")

        # Ensure proper dimensions
        n_state_dims = state.shape[-1]

        # If state has more dimensions than max_state_dim, truncate
        if n_state_dims > self.max_state_dim:
            padded_state = state[:, :self.max_state_dim]
            n_state_dims = self.max_state_dim
        # If state has fewer dimensions than max_state_dim, pad
        elif n_state_dims < self.max_state_dim:
            shape = [dim for dim in state.shape[:-1]] + \
                [self.max_state_dim-n_state_dims]
            shape = torch.Size(shape)
            cat_dim = state.ndim - 1

            padding = torch.zeros(
                shape,
                dtype=state.dtype,
                device=state.device
            )
            padded_state = torch.cat([state, padding], dim=cat_dim)
        else:
            padded_state = state

        # Create mask for real state dimensions
        state_mask = torch.zeros_like(padded_state, dtype=torch.bool)
        state_mask[:, :n_state_dims] = True

        # Number of state tokens
        n_state_tokens = torch.tensor(
            int(padded_state.shape[0]), dtype=torch.int64)

        return padded_state, state_mask, n_state_tokens

    def apply_batch(self, data: Dict[str, torch.Tensor],
                    batch_size: int) -> Dict[str, torch.Tensor]:
        data_split = []
        for i in range(batch_size):
            single_data = {}
            for key, value in data.items():
                single_data[key] = value[i]
            data_split.append(single_data)

        data_split_processed = [self.apply_single(elem) for elem in data_split]

        batch = {}
        for key in data_split_processed[0].keys():
            values = [elem[key] for elem in data_split_processed]
            batch[key] = torch.stack(values)

        return batch

    def apply_single(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process state data for model input in eval mode.

        Args:
            data: Dictionary containing input tensors

        Returns:
            Dictionary with processed tensors
        """
        # Create output dictionary
        result = {}
        for key in data.keys():
            print("export", key, data[key].shape)

        # Process state
        state, state_mask, _ = self.prepare_state(data)
        result["state"] = state
        result["state_mask"] = state_mask

        embodiment_id = self.embodiment_id.detach().clone().to(
            device=state.device)
        # Set embodiment ID
        result["embodiment_id"] = embodiment_id.to(torch.int32)

        return result

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def backward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in data.keys():
            data[key] = data[key].to(torch.float32)
        return data
