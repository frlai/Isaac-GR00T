# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized registry mapping embodiment -> group_name -> individual joint names.

This allows extracting the actual joint element names for each modality key
in GR00T supported embodiments.
"""

from typing import Dict, List, Optional, Union
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag


# =============================================================================
# Joint Registry: embodiment_tag -> modality_type -> group_name -> [joint_names]
# =============================================================================

EMBODIMENT_JOINT_REGISTRY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    # =========================================================================
    # Unitree G1 - Humanoid robot with arms, legs, hands, and waist
    # =========================================================================
    "unitree_g1": {
        "state": {
            "left_leg": [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
            "right_leg": [
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
            "waist": [
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            "left_arm": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            "left_hand": [
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
            ],
            "right_hand": [
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ],
        },
        "action": {
            "left_arm": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            "left_hand": [
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
            ],
            "right_hand": [
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ],
            "waist": [
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            "base_height_command": [
                "base_height",
            ],
            "navigate_command": [
                "nav_vx",
                "nav_vy",
                "nav_vyaw",
            ],
        },
    },

    # =========================================================================
    # Fourier GR1 - Humanoid robot (pretrain embodiment)
    # =========================================================================
    "gr1": {
        "state": {
            "waist": [
                "gr1_waist_yaw",
                "gr1_waist_roll",
                "gr1_waist_pitch",
            ],
            "left_arm": [
                "gr1_left_shoulder_pitch",
                "gr1_left_shoulder_roll",
                "gr1_left_shoulder_yaw",
                "gr1_left_elbow",
                "gr1_left_wrist_roll",
                "gr1_left_wrist_pitch",
                "gr1_left_wrist_yaw",
            ],
            "right_arm": [
                "gr1_right_shoulder_pitch",
                "gr1_right_shoulder_roll",
                "gr1_right_shoulder_yaw",
                "gr1_right_elbow",
                "gr1_right_wrist_roll",
                "gr1_right_wrist_pitch",
                "gr1_right_wrist_yaw",
            ],
            "left_hand": [
                "gr1_left_gripper",
            ],
            "right_hand": [
                "gr1_right_gripper",
            ],
        },
        "action": {
            "waist": [
                "gr1_waist_yaw_cmd",
                "gr1_waist_roll_cmd",
                "gr1_waist_pitch_cmd",
            ],
            "left_arm": [
                "gr1_left_shoulder_pitch_cmd",
                "gr1_left_shoulder_roll_cmd",
                "gr1_left_shoulder_yaw_cmd",
                "gr1_left_elbow_cmd",
                "gr1_left_wrist_roll_cmd",
                "gr1_left_wrist_pitch_cmd",
                "gr1_left_wrist_yaw_cmd",
            ],
            "right_arm": [
                "gr1_right_shoulder_pitch_cmd",
                "gr1_right_shoulder_roll_cmd",
                "gr1_right_shoulder_yaw_cmd",
                "gr1_right_elbow_cmd",
                "gr1_right_wrist_roll_cmd",
                "gr1_right_wrist_pitch_cmd",
                "gr1_right_wrist_yaw_cmd",
            ],
            "left_hand": [
                "gr1_left_gripper_cmd",
            ],
            "right_hand": [
                "gr1_right_gripper_cmd",
            ],
        },
    },

    # =========================================================================
    # LIBERO Panda - Franka Panda arm with EEF control
    # =========================================================================
    "libero_panda": {
        "state": {
            "x": ["eef_x"],
            "y": ["eef_y"],
            "z": ["eef_z"],
            "roll": ["eef_roll"],
            "pitch": ["eef_pitch"],
            "yaw": ["eef_yaw"],
            "gripper": ["gripper_left", "gripper_right"],
        },
        "action": {
            "x": ["delta_x"],
            "y": ["delta_y"],
            "z": ["delta_z"],
            "roll": ["delta_roll"],
            "pitch": ["delta_pitch"],
            "yaw": ["delta_yaw"],
            "gripper": ["gripper_cmd"],
        },
    },

    # =========================================================================
    # OXE WidowX - WidowX robot arm from Open-X-Embodiment
    # =========================================================================
    "oxe_widowx": {
        "state": {
            "x": ["eef_x"],
            "y": ["eef_y"],
            "z": ["eef_z"],
            "roll": ["eef_roll"],
            "pitch": ["eef_pitch"],
            "yaw": ["eef_yaw"],
            "pad": ["pad"],
            "gripper": ["gripper"],
        },
        "action": {
            "x": ["delta_x"],
            "y": ["delta_y"],
            "z": ["delta_z"],
            "roll": ["delta_roll"],
            "pitch": ["delta_pitch"],
            "yaw": ["delta_yaw"],
            "gripper": ["gripper_cmd"],
        },
    },

    # =========================================================================
    # OXE Google - Google robot from Open-X-Embodiment (uses quaternion)
    # =========================================================================
    "oxe_google": {
        "state": {
            "x": ["eef_x"],
            "y": ["eef_y"],
            "z": ["eef_z"],
            "rx": ["quat_x"],
            "ry": ["quat_y"],
            "rz": ["quat_z"],
            "rw": ["quat_w"],
            "gripper": ["gripper"],
        },
        "action": {
            "x": ["delta_x"],
            "y": ["delta_y"],
            "z": ["delta_z"],
            "roll": ["delta_roll"],
            "pitch": ["delta_pitch"],
            "yaw": ["delta_yaw"],
            "gripper": ["gripper_cmd"],
        },
    },

    # =========================================================================
    # OXE DROID - DROID relative EEF + gripper + joint representation.
    # N1.7 uses eef_9d (XYZ + rot6d), gripper_position (1D), joint_position (7D).
    # =========================================================================
    "oxe_droid_relative_eef_relative_joint": {
        "state": {
            "eef_9d": [
                "eef_x",
                "eef_y",
                "eef_z",
                "eef_rot6d_0",
                "eef_rot6d_1",
                "eef_rot6d_2",
                "eef_rot6d_3",
                "eef_rot6d_4",
                "eef_rot6d_5",
            ],
            "gripper_position": ["gripper_position"],
            "joint_position": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
        },
        "action": {
            "eef_9d": [
                "target_eef_x",
                "target_eef_y",
                "target_eef_z",
                "target_eef_rot6d_0",
                "target_eef_rot6d_1",
                "target_eef_rot6d_2",
                "target_eef_rot6d_3",
                "target_eef_rot6d_4",
                "target_eef_rot6d_5",
            ],
            "gripper_position": ["target_gripper_position"],
            "joint_position": [
                "target_joint_0",
                "target_joint_1",
                "target_joint_2",
                "target_joint_3",
                "target_joint_4",
                "target_joint_5",
                "target_joint_6",
            ],
        },
    },

    # =========================================================================
    # Behavior R1 Pro - Humanoid with rich state representation
    # =========================================================================
    "behavior_r1_pro": {
        "state": {
            "robot_pos": ["pos_x", "pos_y", "pos_z"],
            "robot_ori_cos": ["ori_cos_x", "ori_cos_y", "ori_cos_z"],
            "robot_ori_sin": ["ori_sin_x", "ori_sin_y", "ori_sin_z"],
            "robot_2d_ori": ["ori_2d"],
            "robot_2d_ori_cos": ["ori_2d_cos"],
            "robot_2d_ori_sin": ["ori_2d_sin"],
            "robot_lin_vel": ["lin_vel_x", "lin_vel_y", "lin_vel_z"],
            "robot_ang_vel": ["ang_vel_x", "ang_vel_y", "ang_vel_z"],
            "arm_left_qpos": [
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_shoulder_yaw",
                "left_elbow",
                "left_wrist_roll",
                "left_wrist_pitch",
                "left_wrist_yaw",
            ],
            "arm_left_qpos_sin": [
                "left_shoulder_pitch_sin",
                "left_shoulder_roll_sin",
                "left_shoulder_yaw_sin",
                "left_elbow_sin",
                "left_wrist_roll_sin",
                "left_wrist_pitch_sin",
                "left_wrist_yaw_sin",
            ],
            "arm_left_qpos_cos": [
                "left_shoulder_pitch_cos",
                "left_shoulder_roll_cos",
                "left_shoulder_yaw_cos",
                "left_elbow_cos",
                "left_wrist_roll_cos",
                "left_wrist_pitch_cos",
                "left_wrist_yaw_cos",
            ],
            "eef_left_pos": ["eef_left_x", "eef_left_y", "eef_left_z"],
            "eef_left_quat": ["eef_left_qx", "eef_left_qy", "eef_left_qz", "eef_left_qw"],
            "gripper_left_qpos": ["gripper_left_0", "gripper_left_1"],
            "arm_right_qpos": [
                "right_shoulder_pitch",
                "right_shoulder_roll",
                "right_shoulder_yaw",
                "right_elbow",
                "right_wrist_roll",
                "right_wrist_pitch",
                "right_wrist_yaw",
            ],
            "arm_right_qpos_sin": [
                "right_shoulder_pitch_sin",
                "right_shoulder_roll_sin",
                "right_shoulder_yaw_sin",
                "right_elbow_sin",
                "right_wrist_roll_sin",
                "right_wrist_pitch_sin",
                "right_wrist_yaw_sin",
            ],
            "arm_right_qpos_cos": [
                "right_shoulder_pitch_cos",
                "right_shoulder_roll_cos",
                "right_shoulder_yaw_cos",
                "right_elbow_cos",
                "right_wrist_roll_cos",
                "right_wrist_pitch_cos",
                "right_wrist_yaw_cos",
            ],
            "eef_right_pos": ["eef_right_x", "eef_right_y", "eef_right_z"],
            "eef_right_quat": ["eef_right_qx", "eef_right_qy", "eef_right_qz", "eef_right_qw"],
            "gripper_right_qpos": ["gripper_right_0", "gripper_right_1"],
            "trunk_qpos": ["trunk_0", "trunk_1", "trunk_2", "trunk_3"],
        },
        "action": {
            "base": ["base_vx", "base_vy", "base_vyaw"],
            "torso": ["torso_0", "torso_1", "torso_2", "torso_3"],
            "left_arm": [
                "left_shoulder_pitch_cmd",
                "left_shoulder_roll_cmd",
                "left_shoulder_yaw_cmd",
                "left_elbow_cmd",
                "left_wrist_roll_cmd",
                "left_wrist_pitch_cmd",
                "left_wrist_yaw_cmd",
            ],
            "left_gripper": ["left_gripper_cmd"],
            "right_arm": [
                "right_shoulder_pitch_cmd",
                "right_shoulder_roll_cmd",
                "right_shoulder_yaw_cmd",
                "right_elbow_cmd",
                "right_wrist_roll_cmd",
                "right_wrist_pitch_cmd",
                "right_wrist_yaw_cmd",
            ],
            "right_gripper": ["right_gripper_cmd"],
        },
    },
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_state_joint_keys(embodiment_tag: Union[str, EmbodimentTag]) -> List[str]:
    """
    Extract state joint/modality keys (group names) for a given embodiment.
    
    Args:
        embodiment_tag: Either an EmbodimentTag enum or its string value
        
    Returns:
        List of state modality keys (joint group names)
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        embodiment_tag = embodiment_tag.value
    
    if embodiment_tag not in MODALITY_CONFIGS:
        raise ValueError(f"Embodiment '{embodiment_tag}' not found in MODALITY_CONFIGS. "
                        f"Available: {list(MODALITY_CONFIGS.keys())}")
    
    return MODALITY_CONFIGS[embodiment_tag]["state"].modality_keys


def get_action_joint_keys(embodiment_tag: Union[str, EmbodimentTag]) -> List[str]:
    """
    Extract action joint/modality keys (group names) for a given embodiment.
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        embodiment_tag = embodiment_tag.value
    
    if embodiment_tag not in MODALITY_CONFIGS:
        raise ValueError(f"Embodiment '{embodiment_tag}' not found in MODALITY_CONFIGS")
    
    return MODALITY_CONFIGS[embodiment_tag]["action"].modality_keys


def get_all_embodiment_state_keys() -> Dict[str, List[str]]:
    """
    Get state joint keys for all registered embodiments.
    
    Returns:
        Dictionary mapping embodiment tag -> list of state modality keys
    """
    return {
        tag: config["state"].modality_keys 
        for tag, config in MODALITY_CONFIGS.items()
    }


def get_joint_names(
    embodiment_tag: Union[str, EmbodimentTag],
    modality_type: str = "state",
    group_name: Optional[str] = None,
) -> Union[Dict[str, List[str]], List[str]]:
    """
    Get individual joint names for an embodiment.
    
    Args:
        embodiment_tag: The embodiment identifier
        modality_type: "state" or "action"
        group_name: Optional specific group. If None, returns all groups.
        
    Returns:
        If group_name is None: Dict mapping group_name -> [joint_names]
        If group_name is specified: List of joint names for that group
        
    Examples:
        >>> get_joint_names("unitree_g1", "state", "left_arm")
        ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', ...]
        
        >>> get_joint_names("libero_panda", "state")
        {'x': ['eef_x'], 'y': ['eef_y'], ...}
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        embodiment_tag = embodiment_tag.value
    
    if embodiment_tag not in EMBODIMENT_JOINT_REGISTRY:
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not in joint registry. "
            f"Available: {list(EMBODIMENT_JOINT_REGISTRY.keys())}"
        )
    
    embodiment_joints = EMBODIMENT_JOINT_REGISTRY[embodiment_tag]
    
    if modality_type not in embodiment_joints:
        raise ValueError(
            f"Modality type '{modality_type}' not found for {embodiment_tag}. "
            f"Available: {list(embodiment_joints.keys())}"
        )
    
    modality_joints = embodiment_joints[modality_type]
    
    if group_name is None:
        return modality_joints

    if group_name not in modality_joints:
        # Fallback: effort_<x> keys (feed-forward torques) map to the same joints
        # as their position counterpart <x>. Lets users add effort outputs to their
        # modality config without duplicating joint-name registrations.
        if group_name.startswith("effort_"):
            base_key = group_name[len("effort_"):]
            if base_key in modality_joints:
                return modality_joints[base_key]

        raise ValueError(
            f"Group '{group_name}' not found in {embodiment_tag}/{modality_type}. "
            f"Available: {list(modality_joints.keys())}"
        )

    return modality_joints[group_name]


def register_embodiment_joints(
    embodiment_tag: Union[str, EmbodimentTag],
    joints: Dict[str, Dict[str, List[str]]],
    base_embodiment: Optional[Union[str, EmbodimentTag]] = None,
) -> None:
    """Register a new embodiment in the joint registry.

    This is the programmatic equivalent of ``register_modality_config`` in GR00T.
    Use it to support fine-tunes with ``EmbodimentTag.NEW_EMBODIMENT`` or custom
    embodiments not in the pre-registered registry.

    Args:
        embodiment_tag: The embodiment identifier to register.
        joints: Mapping of ``modality_type`` ("state"/"action") to a mapping of
            ``group_name`` to a list of joint names.  Only keys you supply are
            stored; merging with a base embodiment is controlled via
            ``base_embodiment``.
        base_embodiment: Optional embodiment to copy joints from before applying
            overrides.  Useful when a custom embodiment is a variant of an
            existing one (e.g. a different whole-body controller on the same
            robot).

    Example::

        register_embodiment_joints(
            EmbodimentTag.NEW_EMBODIMENT,
            joints={},  # use everything from unitree_g1
            base_embodiment="unitree_g1",
        )
    """
    import copy as _copy

    if isinstance(embodiment_tag, EmbodimentTag):
        embodiment_tag = embodiment_tag.value

    if base_embodiment is not None:
        if isinstance(base_embodiment, EmbodimentTag):
            base_embodiment = base_embodiment.value
        if base_embodiment not in EMBODIMENT_JOINT_REGISTRY:
            raise ValueError(
                f"Base embodiment '{base_embodiment}' not in registry. "
                f"Available: {list(EMBODIMENT_JOINT_REGISTRY.keys())}"
            )
        merged = _copy.deepcopy(EMBODIMENT_JOINT_REGISTRY[base_embodiment])
    else:
        merged = {}

    for modality_type, groups in joints.items():
        merged.setdefault(modality_type, {}).update(groups)

    EMBODIMENT_JOINT_REGISTRY[embodiment_tag] = merged


def get_flat_joint_names(
    embodiment_tag: Union[str, EmbodimentTag],
    modality_type: str = "state",
) -> List[str]:
    """
    Get all joint names flattened in order (matching tensor element order).
    
    The order follows the modality_keys defined in MODALITY_CONFIGS.
    
    Args:
        embodiment_tag: The embodiment identifier
        modality_type: "state" or "action"
        
    Returns:
        Flat list of all joint names in tensor order
        
    Example:
        >>> get_flat_joint_names("unitree_g1", "state")
        ['left_hip_pitch_joint', 'left_hip_roll_joint', ..., 'right_hand_thumb_2_joint']
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        tag_val = embodiment_tag.value
    else:
        tag_val = embodiment_tag
    
    joints_by_group = get_joint_names(tag_val, modality_type)
    
    if tag_val not in MODALITY_CONFIGS:
        raise ValueError(f"Embodiment '{tag_val}' not found in MODALITY_CONFIGS")
    
    ordered_keys = MODALITY_CONFIGS[tag_val][modality_type].modality_keys
    
    flat_joints = []
    for key in ordered_keys:
        if key in joints_by_group:
            flat_joints.extend(joints_by_group[key])
        else:
            raise ValueError(
                f"Group '{key}' from MODALITY_CONFIGS not found in joint registry for {tag_val}"
            )
    return flat_joints


def get_joint_index_mapping(
    embodiment_tag: Union[str, EmbodimentTag],
    modality_type: str = "state",
) -> Dict[str, int]:
    """
    Get a mapping from joint name to its index in the flat tensor.
    
    Args:
        embodiment_tag: The embodiment identifier
        modality_type: "state" or "action"
        
    Returns:
        Dict mapping joint_name -> index in tensor
        
    Example:
        >>> mapping = get_joint_index_mapping("unitree_g1", "state")
        >>> mapping["left_elbow_joint"]
        15
    """
    flat_joints = get_flat_joint_names(embodiment_tag, modality_type)
    return {name: idx for idx, name in enumerate(flat_joints)}


def get_group_index_ranges(
    embodiment_tag: Union[str, EmbodimentTag],
    modality_type: str = "state",
) -> Dict[str, tuple]:
    """
    Get the start and end indices for each group in the flat tensor.
    
    Args:
        embodiment_tag: The embodiment identifier
        modality_type: "state" or "action"
        
    Returns:
        Dict mapping group_name -> (start_idx, end_idx)
        
    Example:
        >>> ranges = get_group_index_ranges("unitree_g1", "state")
        >>> ranges["left_arm"]
        (15, 22)  # 7 joints
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        tag_val = embodiment_tag.value
    else:
        tag_val = embodiment_tag
    
    joints_by_group = get_joint_names(tag_val, modality_type)
    ordered_keys = MODALITY_CONFIGS[tag_val][modality_type].modality_keys
    
    ranges = {}
    current_idx = 0
    for key in ordered_keys:
        if key in joints_by_group:
            num_joints = len(joints_by_group[key])
            ranges[key] = (current_idx, current_idx + num_joints)
            current_idx += num_joints
    
    return ranges


def get_embodiment_summary(embodiment_tag: Union[str, EmbodimentTag]) -> Dict:
    """
    Get a complete summary of an embodiment's joint configuration.
    
    Args:
        embodiment_tag: The embodiment identifier
        
    Returns:
        Dict with full summary including state/action groups, joint names, and dimensions
    """
    if isinstance(embodiment_tag, EmbodimentTag):
        tag_val = embodiment_tag.value
    else:
        tag_val = embodiment_tag
    
    summary = {
        "embodiment": tag_val,
        "state": {
            "groups": get_state_joint_keys(tag_val),
            "total_dim": len(get_flat_joint_names(tag_val, "state")),
            "group_dims": {},
            "group_joints": {},
        },
        "action": {
            "groups": get_action_joint_keys(tag_val),
            "total_dim": len(get_flat_joint_names(tag_val, "action")),
            "group_dims": {},
            "group_joints": {},
        },
    }
    
    for modality in ["state", "action"]:
        joints_by_group = get_joint_names(tag_val, modality)
        for group in summary[modality]["groups"]:
            if group in joints_by_group:
                summary[modality]["group_dims"][group] = len(joints_by_group[group])
                summary[modality]["group_joints"][group] = joints_by_group[group]
    
    return summary


def list_registered_embodiments() -> List[str]:
    """List all embodiments with joint registrations."""
    return list(EMBODIMENT_JOINT_REGISTRY.keys())


# =============================================================================
# Main - Demo usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GR00T Embodiment Joint Registry Demo")
    print("=" * 80)
    
    print("\n1. Registered embodiments with joint mappings:")
    print(f"   {list_registered_embodiments()}")
    
    print("\n2. Unitree G1 - State joint groups:")
    for group in get_state_joint_keys("unitree_g1"):
        joints = get_joint_names("unitree_g1", "state", group)
        print(f"   {group}: {len(joints)} joints")
        for j in joints:
            print(f"      - {j}")
    
    print("\n3. Unitree G1 - Group index ranges (state):")
    ranges = get_group_index_ranges("unitree_g1", "state")
    for group, (start, end) in ranges.items():
        print(f"   {group}: indices [{start}, {end}) - {end - start} elements")
    
    print("\n4. LIBERO Panda - Flat joint names (state):")
    flat = get_flat_joint_names("libero_panda", "state")
    print(f"   {flat}")
    
    print("\n5. Joint name to index mapping (unitree_g1 action):")
    mapping = get_joint_index_mapping("unitree_g1", "action")
    print(f"   left_elbow_joint -> index {mapping.get('left_elbow_joint', 'N/A')}")
    print(f"   right_shoulder_pitch_joint -> index {mapping.get('right_shoulder_pitch_joint', 'N/A')}")
    
    print("\n6. Full embodiment summary (behavior_r1_pro):")
    summary = get_embodiment_summary("behavior_r1_pro")
    print(f"   State: {summary['state']['total_dim']} dims across {len(summary['state']['groups'])} groups")
    print(f"   Action: {summary['action']['total_dim']} dims across {len(summary['action']['groups'])} groups")
    
    print("\n" + "=" * 80)
    print("Done!")
