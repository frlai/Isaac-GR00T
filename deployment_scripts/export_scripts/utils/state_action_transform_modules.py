import torch
import torch.nn as nn
from typing import List, Dict, Any
import numpy as np
import functools
import pytorch3d.transforms as pt


class RotationTransform(nn.Module):
    valid_reps = ["axis_angle", "euler_angles",
                  "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace(
                "r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace(
                "r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"),
                     getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(
                    func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(
                pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(
                    func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    def apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_funcs(x, self.inverse_funcs)


class Normalizer(nn.Module):
    valid_modes = ["q99", "mean_std", "min_max", "binary", "scale"]

    def __init__(self, mode: str, statistics: dict):
        """Initialize a normalizer with a specific normalization mode and statistics.

        Args:
            mode: Normalization mode to use
            statistics: Statistics required for the normalization
        """
        super().__init__()
        self.mode = mode
        assert mode in self.valid_modes, f"Invalid normalization mode: {mode}."\
            f"Valid modes are {self.valid_modes}"

        # Convert statistics to tensors and store them
        self.statistics = {}
        for key, value in statistics.items():
            self.statistics[key] = value.clone().detach()

        # Preselect the normalization function based on mode
        if mode == "q99":
            self._normalize_fn = self._normalize_q99
            self._inverse_normalize_fn = self._inverse_normalize_q99
        elif mode == "mean_std":
            self._normalize_fn = self._normalize_mean_std
            self._inverse_normalize_fn = self._inverse_normalize_mean_std
        elif mode == "min_max":
            self._normalize_fn = self._normalize_min_max
            self._inverse_normalize_fn = self._inverse_normalize_min_max
        elif mode == "scale":
            self._normalize_fn = self._normalize_scale
            self._inverse_normalize_fn = self._inverse_normalize_scale
        elif mode == "binary":
            self._normalize_fn = self._normalize_binary
            self._inverse_normalize_fn = self._inverse_normalize_binary
        else:
            raise ValueError(f"Invalid normalization mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input tensor.

        Args:
            x: Input tensor to normalize

        Returns:
            Normalized tensor
        """
        return self._normalize_fn(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse normalization to input tensor.

        Args:
            x: Normalized tensor

        Returns:
            Tensor in original scale
        """
        return self._inverse_normalize_fn(x)

    def _normalize_q99(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using q99 method.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor in range [-1, 1]
        """
        # Range of q99 is [-1, 1]
        q01 = self.statistics["q01"].to(x.dtype)
        q99 = self.statistics["q99"].to(x.dtype)

        # In the case of q01 == q99, the normalization will be undefined
        # So we set the normalized values to the original values
        mask = q01 != q99
        normalized = torch.zeros_like(x, device=x.device)

        # Normalize the values where q01 != q99
        # Formula: 2 * (x - q01) / (q99 - q01) - 1
        normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
            q99[..., mask] - q01[..., mask]
        )
        normalized[..., mask] = 2 * normalized[..., mask] - 1

        # Set the normalized values to the original values where q01 == q99
        normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        # Clip the normalized values to be between -1 and 1
        normalized = torch.clamp(normalized, -1, 1)

        return normalized

    def _normalize_mean_std(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using mean and standard deviation.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor with zero mean and unit variance
        """
        # Range of mean_std is not fixed, but can be positive or negative
        mean = self.statistics["mean"].to(device=x.device, dtype=x.dtype)
        std = self.statistics["std"].to(device=x.device, dtype=x.dtype)

        # In the case of std == 0, the normalization will be undefined
        # So we set the normalized values to the original values
        mask = std != 0
        normalized = torch.zeros_like(x, device=x.device)

        # Normalize the values where std != 0
        # Formula: (x - mean) / std
        normalized[..., mask] = (
            x[..., mask] - mean[..., mask]) / std[..., mask]

        # Set the normalized values to the original values where std == 0
        normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        return normalized

    def _normalize_min_max(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using min-max scaling.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor in range [-1, 1]
        """
        # Range of min_max is [-1, 1]
        min_val = self.statistics["min"].to(device=x.device, dtype=x.dtype)
        max_val = self.statistics["max"].to(device=x.device, dtype=x.dtype)
        # In the case of min == max, the normalization will be undefined
        # So we set the normalized values to 0
        mask = min_val != max_val
        normalized = torch.zeros_like(x, device=x.device)

        # Normalize the values where min != max
        # Formula: 2 * (x - min) / (max - min) - 1
        normalized[..., mask] = (x[..., mask] - min_val[..., mask]) / (
            max_val[..., mask] - min_val[..., mask]
        )
        normalized[..., mask] = 2 * normalized[..., mask] - 1

        # Set the normalized values to 0 where min == max
        normalized[..., ~mask] = 0

        return normalized

    def _normalize_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by scaling by the absolute maximum value.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor in range [0, 1]
        """
        # Range of scale is [0, 1]
        min_val = self.statistics["min"].to(device=x.device, dtype=x.dtype)
        max_val = self.statistics["max"].to(device=x.device, dtype=x.dtype)
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))

        mask = abs_max != 0
        normalized = torch.zeros_like(x, device=x.device)
        normalized[..., mask] = x[..., mask] / abs_max[..., mask]
        normalized[..., ~mask] = 0

        return normalized

    def _normalize_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Binarize the input tensor.

        Args:
            x: Input tensor

        Returns:
            Binary tensor (0 or 1)
        """
        # Range of binary is [0, 1]
        return (x > 0.5).to(device=x.device, dtype=x.dtype)

    def _inverse_normalize_q99(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of q99 normalization.

        Args:
            x: Normalized tensor in range [-1, 1]

        Returns:
            Tensor in original scale
        """
        q01 = self.statistics["q01"].to(device=x.device, dtype=x.dtype)
        q99 = self.statistics["q99"].to(device=x.device, dtype=x.dtype)
        return (x + 1) / 2 * (q99 - q01) + q01

    def _inverse_normalize_mean_std(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of mean-std normalization.

        Args:
            x: Normalized tensor with zero mean and unit variance

        Returns:
            Tensor in original scale
        """
        mean = self.statistics["mean"].to(device=x.device, dtype=x.dtype)
        std = self.statistics["std"].to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _inverse_normalize_min_max(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of min-max normalization.

        Args:
            x: Normalized tensor in range [-1, 1]

        Returns:
            Tensor in original scale
        """
        min_val = self.statistics["min"].to(device=x.device, dtype=x.dtype)
        max_val = self.statistics["max"].to(device=x.device, dtype=x.dtype)
        return (x + 1) / 2 * (max_val - min_val) + min_val

    def _inverse_normalize_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of scale normalization.

        Args:
            x: Normalized tensor in range [0, 1]

        Returns:
            Tensor in original scale
        """
        min_val = self.statistics["min"].to(device=x.device, dtype=x.dtype)
        max_val = self.statistics["max"].to(device=x.device, dtype=x.dtype)
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
        return x * abs_max

    def _inverse_normalize_binary(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of binary transformation (which is itself, as it's a threshold operation).

        Args:
            x: Binary tensor (0 or 1)

        Returns:
            Binary tensor (0 or 1)
        """
        return (x > 0.5).to(device=x.device, dtype=x.dtype)


class StateActionToTensor(nn.Module):
    """Exportable version of StateActionToTensor."""

    def __init__(self, apply_to: List[str],
                 output_dtypes: Dict[str, torch.dtype] = None,
                 input_dtypes: Dict[str, torch.dtype] = None,
                 backward: bool = False, **kwargs):
        """Initialize the StateActionToTensor module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            output_dtypes: Dictionary mapping keys to output dtypes
            backward: Whether to use the backward method as the forward method
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        self.output_dtypes = output_dtypes or {}
        self.input_dtypes = input_dtypes or {}

        # If backward flag is True, replace forward with backward method
        if backward:
            self.forward = self.backward

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Convert specified tensors in the dictionary to specified dtypes.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with converted tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply dtype conversion to each key in apply_to
        for key in self.apply_to:
            if key in result:
                # Convert to specified dtype if provided
                if key in self.output_dtypes:
                    result[key] = result[key].to(self.output_dtypes[key])

        return result

    def backward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return data


class StateActionPerturbation(nn.Module):
    """Exportable version of StateActionPerturbation."""

    def __init__(self, apply_to: List[str], std: float, backward: bool = False, **kwargs):
        """Initialize the StateActionPerturbation module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            std: Standard deviation of the noise to add
            backward: Whether to use the backward method as the forward method
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        self.std = std

        # If backward flag is True, replace forward with backward method
        # backward method is the same as the forward method
        if backward:
            pass

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Add perturbation to specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with perturbed tensors
        """
        # Since training is always False, we don't need to add perturbation
        return data


class StateActionTransform(nn.Module):
    """Exportable version of StateActionTransform."""

    def __init__(self,
                 apply_to: List[str],
                 normalization_modes: Dict[str, str] = None,
                 target_rotations: Dict[str, str] = None,
                 normalization_statistics: Dict[str, Dict] = None,
                 modality_metadata: Dict[str, Any] = None,
                 backward: bool = False,
                 **kwargs):
        """Initialize the StateActionTransform module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            normalization_modes: Dictionary mapping keys to normalization modes
            target_rotations: Dictionary mapping keys to target rotation representations
            normalization_statistics: Dictionary mapping keys to normalization statistics
            modality_metadata: Dictionary mapping keys to modality metadata
            backward: Whether to use the backward method as the forward method
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        self.normalization_modes = normalization_modes or {}
        self.target_rotations = target_rotations or {}
        self.normalization_statistics = normalization_statistics or {}
        self.modality_metadata = modality_metadata or {}

        if backward:
            self.forward = self.backward

        # Default min-max statistics for different rotation representations
        self._DEFAULT_MIN_MAX_STATISTICS = {
            "rotation_6d": {
                "min": [-1, -1, -1, -1, -1, -1],
                "max": [1, 1, 1, 1, 1, 1],
            },
            "euler_angles": {
                "min": [-np.pi, -np.pi, -np.pi],
                "max": [np.pi, np.pi, np.pi],
            },
            "quaternion": {
                "min": [-1, -1, -1, -1],
                "max": [1, 1, 1, 1],
            },
            "axis_angle": {
                "min": [-np.pi, -np.pi, -np.pi],
                "max": [np.pi, np.pi, np.pi],
            },
        }

        # Store transform functions for each key
        self._transform_fns = {}

        # Initialize rotation transformers and normalizers
        self._rotation_transformers = {}
        self._normalizers = {}

        # Set up rotation transformers
        self._setup_rotation_transformers()

        # Set up normalizers
        self._setup_normalizers()

        # For each key in apply_to, set up a transform function
        for key in self.apply_to:
            self._setup_transform_for_key(key)

    def _setup_rotation_transformers(self):
        """Set up rotation transformers for keys with target rotations."""
        for key in self.target_rotations:
            if key not in self.modality_metadata:
                continue

            # Get source rotation type
            from_rep = self.modality_metadata[key].rotation_type
            if from_rep is None:
                continue

            # Get target rotation type
            to_rep = self.target_rotations[key]

            # If different, create a rotation transformer
            if from_rep.value != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep
                )

    def _setup_normalizers(self):
        """Set up normalizers for keys with normalization modes."""
        for key in self.normalization_modes:
            if key not in self.apply_to:
                continue

            mode = self.normalization_modes[key]

            # Handle rotation normalizations specially
            if key in self._rotation_transformers and mode == "min_max":
                if self.modality_metadata[key].absolute:
                    # Get rotation type
                    rotation_type = self.target_rotations[key]
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"

                    # Use default statistics for rotation type
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                else:
                    # Skip relative rotations that need normalization
                    continue
            else:
                # Use provided statistics for this key
                if key in self.normalization_statistics:
                    statistics = self.normalization_statistics[key]
                else:
                    continue

                # Validate continuous vs. binary normalization
                if (key in self.modality_metadata
                    and not self.modality_metadata[key].continuous
                        and mode != "binary"):
                    # Default to binary for non-continuous data
                    mode = "binary"

            # Create the normalizer
            self._normalizers[key] = Normalizer(
                mode=mode, statistics=statistics)

    def _setup_transform_for_key(self, key: str):
        """Set up the transformation function for a specific key."""
        # Define a function that applies the appropriate transforms to this key
        def transform_fn(x: torch.Tensor) -> torch.Tensor:
            # Start with the input tensor
            result = x

            # Apply rotation transform if needed
            if key in self._rotation_transformers:
                result = self._rotation_transformers[key].transform(result)

            # Apply normalization if needed
            if key in self._normalizers:
                result = self._normalizers[key].forward(result)

            return result

        # Store the transform function for this key
        self._transform_fns[key] = transform_fn

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply transformations to specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with transformed tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply the pre-determined transform function for each key
        for key in self.apply_to:
            if key in result and key in self._transform_fns:
                result[key] = self._transform_fns[key](result[key])

        return result

    def backward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply inverse transformations to get back to original representations.

        Args:
            data: Dictionary of transformed tensors

        Returns:
            Dictionary with tensors in original representations
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply inverse transformations in reverse order
        for key in self.apply_to:
            if key in result:
                # Apply inverse normalization if needed
                if key in self._normalizers:
                    result[key] = self._normalizers[key].inverse(result[key])

                # Apply inverse rotation transform if needed
                if key in self._rotation_transformers:
                    result[key] = self._rotation_transformers[key].inverse(
                        result[key])
        return result


class StateActionDropout(nn.Module):
    """Exportable version of StateActionDropout."""

    def __init__(self, apply_to: List[str], dropout_prob: float, **kwargs):
        """Initialize the StateActionDropout module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            dropout_prob: Probability of dropping out tensor values
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to
        self.dropout_prob = dropout_prob

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply dropout to specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with dropout applied to tensors
        """
        # Since training is always False, we don't need to apply dropout
        return data


class StateActionSinCosTransform(nn.Module):
    """Exportable version of StateActionSinCosTransform."""

    def __init__(self, apply_to: List[str], **kwargs):
        """Initialize the StateActionSinCosTransform module.

        Args:
            apply_to: List of keys in the input dictionary to apply the transform to
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.apply_to = apply_to

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Apply sin-cos transformation to specified tensors in the dictionary.

        Args:
            data: Dictionary of tensors
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with sin-cos transformed tensors
        """
        # Create a new dictionary to avoid modifying the input
        result = dict(data)

        # Apply sin-cos transformation to each key in apply_to
        for key in self.apply_to:
            if key in result:
                sin_state = torch.sin(result[key])
                cos_state = torch.cos(result[key])
                result[key] = torch.cat([sin_state, cos_state], dim=-1)

        return result
