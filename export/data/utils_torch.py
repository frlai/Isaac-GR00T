# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Torch versions of normalization utilities for ONNX export.

These mirror the numpy functions in gr00t/data/utils.py but use torch operations
so they can be traced/exported to ONNX.
"""

import torch
from torch import Tensor


def apply_sin_cos_encoding(values: Tensor) -> Tensor:
    """Apply sin/cos encoding to values.

    Args:
        values: Tensor of shape (..., D) containing values to encode

    Returns:
        Tensor of shape (..., 2*D) with [sin, cos] concatenated

    Note: This DOUBLES the dimension. For example:
        Input:  [v₁, v₂, v₃] with shape (..., 3)
        Output: [sin(v₁), sin(v₂), sin(v₃), cos(v₁), cos(v₂), cos(v₃)] with shape (..., 6)
    """
    sin_values = torch.sin(values)
    cos_values = torch.cos(values)
    # Concatenate sin and cos: [sin(v1), sin(v2), ..., cos(v1), cos(v2), ...]
    return torch.cat([sin_values, cos_values], dim=-1)


def normalize_values_minmax(values: Tensor, params: dict) -> Tensor:
    """
    Normalize values using min-max normalization to [-1, 1] range.

    Args:
        values: Input tensor to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
        params: Dictionary with "min" and "max" keys
            - params["min"]: Minimum values for normalization
            - params["max"]: Maximum values for normalization

    Returns:
        Normalized values in [-1, 1] range
            - Same shape as input values
            - Values are linearly mapped from [min, max] to [-1, 1]
            - For features where min == max, normalized value is 0
    """
    min_vals = params["min"]
    max_vals = params["max"]
    
    # Convert to tensor if numpy
    if not isinstance(min_vals, Tensor):
        min_vals = torch.tensor(min_vals, dtype=values.dtype, device=values.device)
    if not isinstance(max_vals, Tensor):
        max_vals = torch.tensor(max_vals, dtype=values.dtype, device=values.device)
    
    # Create mask for non-constant features (where max != min)
    mask = ~torch.isclose(max_vals, min_vals)
    
    # Compute range with safe minimum to avoid division by zero
    range_vals = max_vals - min_vals
    safe_range = torch.clamp(range_vals, min=1e-8)
    
    # Normalize: map to [0, 1], then to [-1, 1]
    full_normalized = 2 * (values - min_vals) / safe_range - 1
    
    # For constant features (max == min), output zero
    normalized = torch.where(mask.expand_as(values), full_normalized, torch.zeros_like(values))
    
    return normalized


def unnormalize_values_minmax(normalized_values: Tensor, params: dict) -> Tensor:
    """
    Min-max unnormalization from [-1, 1] range back to original range.

    Args:
        normalized_values: Normalized input tensor in [-1, 1] range
            - Shape: (T, D) or (B, T, D)
        params: Dictionary with "min" and "max" keys

    Returns:
        Unnormalized values in original range [min, max]
            - Same shape as input
            - Input values are clipped to [-1, 1] before unnormalization
    """
    min_vals = params["min"]
    max_vals = params["max"]
    
    # Convert to tensor if numpy
    if not isinstance(min_vals, Tensor):
        min_vals = torch.tensor(min_vals, dtype=normalized_values.dtype, device=normalized_values.device)
    if not isinstance(max_vals, Tensor):
        max_vals = torch.tensor(max_vals, dtype=normalized_values.dtype, device=normalized_values.device)
    
    range_vals = max_vals - min_vals
    
    # Unnormalize from [-1, 1]: x = (normalized + 1) / 2 * range + min
    unnormalized = (torch.clamp(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def normalize_values_meanstd(values: Tensor, params: dict) -> Tensor:
    """
    Normalize values using mean-std (z-score) normalization.

    Args:
        values: Input tensor to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
        params: Dictionary with "mean" and "std" keys

    Returns:
        Normalized values using z-score normalization
            - Same shape as input values
            - Values are transformed as: (x - mean) / std
            - For features where std == 0, normalized value equals original value
    """
    mean_vals = params["mean"]
    std_vals = params["std"]
    
    # Convert to tensor if numpy
    if not isinstance(mean_vals, Tensor):
        mean_vals = torch.tensor(mean_vals, dtype=values.dtype, device=values.device)
    if not isinstance(std_vals, Tensor):
        std_vals = torch.tensor(std_vals, dtype=values.dtype, device=values.device)
    
    # Create mask for non-zero standard deviations
    mask = std_vals != 0
    
    # Compute z-score normalization with safe division
    safe_std = torch.clamp(std_vals, min=1e-8)
    normalized = (values - mean_vals) / safe_std
    
    # For zero-std features, keep original values
    normalized = torch.where(mask.expand_as(values), normalized, values)
    
    return normalized


def unnormalize_values_meanstd(normalized_values: Tensor, params: dict) -> Tensor:
    """
    Mean-std unnormalization (reverse z-score normalization).

    Args:
        normalized_values: Normalized input tensor (z-scores)
            - Shape: (T, D) or (B, T, D)
        params: Dictionary with "mean" and "std" keys

    Returns:
        Unnormalized values in original scale
            - Same shape as input
            - Values are transformed as: x * std + mean
            - For features where std == 0, unnormalized value equals normalized value
    """
    mean_vals = params["mean"]
    std_vals = params["std"]
    
    # Convert to tensor if numpy
    if not isinstance(mean_vals, Tensor):
        mean_vals = torch.tensor(mean_vals, dtype=normalized_values.dtype, device=normalized_values.device)
    if not isinstance(std_vals, Tensor):
        std_vals = torch.tensor(std_vals, dtype=normalized_values.dtype, device=normalized_values.device)
    
    # Create mask for non-zero standard deviations
    mask = std_vals != 0
    
    # Unnormalize: x = normalized * std + mean
    unnormalized = normalized_values * std_vals + mean_vals
    
    # For zero-std features, keep normalized values
    unnormalized = torch.where(mask.expand_as(normalized_values), unnormalized, normalized_values)
    
    return unnormalized


# =============================================================================
# Rotation and Homogeneous Matrix Utilities (for action conversion)
# =============================================================================

def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    
    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix.
    
    Args:
        rot6d: (..., 6) tensor - first two rows of rotation matrix flattened
        
    Returns:
        Rotation matrix (..., 3, 3)
    """
    # Reshape to (..., 2, 3)
    shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(*shape, 2, 3)
    
    # First two rows
    row1 = rot6d[..., 0, :]  # (..., 3)
    row2 = rot6d[..., 1, :]  # (..., 3)
    
    # Normalize first row
    row1 = row1 / torch.linalg.norm(row1, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Gram-Schmidt orthogonalization for second row
    dot = (row1 * row2).sum(dim=-1, keepdim=True)
    row2 = row2 - dot * row1
    row2 = row2 / torch.linalg.norm(row2, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Third row is cross product
    row3 = torch.cross(row1, row2, dim=-1)
    
    # Stack to rotation matrix (..., 3, 3)
    return torch.stack([row1, row2, row3], dim=-2)


def matrix_to_rot6d(rotation_matrix: Tensor) -> Tensor:
    """
    Convert rotation matrix to 6D rotation representation.
    
    Args:
        rotation_matrix: (..., 3, 3) rotation matrix
        
    Returns:
        6D rotation (..., 6) - first two rows flattened
    """
    # Take first two rows and flatten
    return rotation_matrix[..., :2, :].reshape(*rotation_matrix.shape[:-2], 6)


def xyz_rot6d_to_homogeneous(xyz_rot6d: Tensor) -> Tensor:
    """
    Convert xyz + rot6d (9,) to homogeneous transformation matrix (4, 4).
    
    Args:
        xyz_rot6d: (..., 9) tensor with [x, y, z, r1, r2, r3, r4, r5, r6]
        
    Returns:
        Homogeneous matrix (..., 4, 4)
    """
    shape = xyz_rot6d.shape[:-1]
    xyz = xyz_rot6d[..., :3]
    rot6d = xyz_rot6d[..., 3:]
    
    rot_matrix = rot6d_to_matrix(rot6d)  # (..., 3, 3)
    
    # Build homogeneous matrix functionally so leapp tracing is preserved.
    top = torch.cat([rot_matrix, xyz.unsqueeze(-1)], dim=-1)
    zeros = torch.zeros_like(xyz[..., :1])
    ones = torch.ones_like(zeros)
    bottom = torch.cat([zeros, zeros, zeros, ones], dim=-1).unsqueeze(-2)
    return torch.cat([top, bottom], dim=-2)


def homogeneous_to_xyz_rot6d(H: Tensor) -> Tensor:
    """
    Convert homogeneous matrix (4, 4) to xyz + rot6d (9,).
    
    Args:
        H: (..., 4, 4) homogeneous transformation matrix
        
    Returns:
        (..., 9) tensor with [x, y, z, r1, r2, r3, r4, r5, r6]
    """
    xyz = H[..., :3, 3]
    rot_matrix = H[..., :3, :3]
    rot6d = matrix_to_rot6d(rot_matrix)
    
    return torch.cat([xyz, rot6d], dim=-1)


def invert_homogeneous(T: Tensor) -> Tensor:
    """
    Invert a homogeneous transformation matrix.
    
    For a rigid transformation T = [R|t; 0|1], the inverse is [R^T|-R^T*t; 0|1].
    
    Args:
        T: (..., 4, 4) homogeneous transformation matrix
        
    Returns:
        Inverse transformation (..., 4, 4)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    # R_inv = R^T (rotation matrices are orthogonal)
    R_inv = R.transpose(-2, -1)
    
    # t_inv = -R_inv @ t
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
    
    # Build inverse functionally so traced tensors are not lost through partial assignment.
    top = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)
    zeros = torch.zeros_like(t_inv[..., :1])
    ones = torch.ones_like(zeros)
    bottom = torch.cat([zeros, zeros, zeros, ones], dim=-1).unsqueeze(-2)
    return torch.cat([top, bottom], dim=-2)


def convert_to_absolute_action_eef(
    action: Tensor,
    reference_state: Tensor,
) -> Tensor:
    """
    Convert relative end-effector action to absolute action.
    
    Performs homogeneous matrix composition: T_absolute = T_reference @ T_relative
    
    Args:
        action: (T, 9) relative action tensor (xyz + rot6d)
        reference_state: (9,) reference state tensor (xyz + rot6d)
        
    Returns:
        (T, 9) absolute action tensor
    """
    # Convert reference state to homogeneous matrix
    T_ref = xyz_rot6d_to_homogeneous(reference_state)  # (4, 4)
    
    # Convert each action to homogeneous
    T_relative = xyz_rot6d_to_homogeneous(action)  # (T, 4, 4)
    
    # T_absolute = T_ref @ T_relative for each timestep
    T_absolute = torch.matmul(T_ref.unsqueeze(0), T_relative)  # (T, 4, 4)
    
    # Convert back to xyz + rot6d
    return homogeneous_to_xyz_rot6d(T_absolute)  # (T, 9)


def convert_to_absolute_action_joints(
    action: Tensor,
    reference_state: Tensor,
) -> Tensor:
    """
    Convert relative joint action to absolute action.
    
    For joints, this is simple addition: absolute = reference + relative
    
    Args:
        action: (T, D) relative action tensor
        reference_state: (D,) reference state tensor
        
    Returns:
        (T, D) absolute action tensor
    """
    return reference_state.unsqueeze(0) + action
