# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import importlib.util
import torch
from gr00t.data.state_action.state_action_processor import StateActionProcessor

from gr00t.configs.data.embodiment_configs import (
    ActionRepresentation,
    ActionType,
)

# Load utils_torch from the same directory (works with importlib loading)
_utils_path = os.path.join(os.path.dirname(__file__), "utils_torch.py")
_spec = importlib.util.spec_from_file_location("utils_torch", _utils_path)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)
apply_sin_cos_encoding = _utils_module.apply_sin_cos_encoding
normalize_values_meanstd = _utils_module.normalize_values_meanstd
normalize_values_minmax = _utils_module.normalize_values_minmax
unnormalize_values_meanstd = _utils_module.unnormalize_values_meanstd
unnormalize_values_minmax = _utils_module.unnormalize_values_minmax
convert_to_absolute_action_eef = _utils_module.convert_to_absolute_action_eef
convert_to_absolute_action_joints = _utils_module.convert_to_absolute_action_joints


# TODO: change the way this works to be more like how the original works.

class StateActionProcessorTorch:
    ''' optimized version of StateActionProcessor for torch when performing inference'''
    def __init__(self, state_action_processor: StateActionProcessor):
        self.clip_outliers = state_action_processor.clip_outliers
        self.use_relative_action = state_action_processor.use_relative_action
        self.apply_sincos_state_encoding = state_action_processor.apply_sincos_state_encoding

        self.norm_params = state_action_processor.norm_params
        self.modality_configs = state_action_processor.modality_configs


    def apply(self,
        state: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
        embodiment_tag: str,
        **kwargs
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        processed_state = self.apply_state(state, embodiment_tag)

        return processed_state, {}

    def apply_state(self,
        state: dict[str, torch.Tensor],
        embodiment_tag: str,
    ) -> dict[str, torch.Tensor]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys


        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])
            
            # Strategy 2: Mean/std normalization
            elif (
                hasattr(self.modality_configs[embodiment_tag]["state"], "mean_std_embedding_keys")
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = torch.clamp(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

        return normalized_values
    

    def unapply_action(self,
        action: dict[str, torch.Tensor],
        embodiment_tag: str,
        state: dict[str, torch.Tensor] | None = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (if needed)
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )
                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key
                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state.unsqueeze(0)
                        relative_action = relative_action.unsqueeze(0)
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state.unsqueeze(0)

                    # Convert batched relative actions to absolute
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action):
                        # Use last timestep of state as reference
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1],
                            action_type=action_config.type,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = torch.stack(absolute_actions, dim=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def _convert_to_absolute_action(
        self,
        action: torch.Tensor,
        reference_state: torch.Tensor,
        action_type: ActionType,
    ) -> torch.Tensor:
        """
        Convert relative action to absolute action using reference state.
        
        This is the torch-traceable version of StateActionProcessor._convert_to_absolute_action.
        
        Args:
            action: (T, D) relative action tensor
            reference_state: (D,) reference state tensor
            action_type: ActionType.EEF (9D: xyz + rot6d) or ActionType.NON_EEF (joints)
            
        Returns:
            (T, D) absolute action tensor
        """
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            assert action.shape[1] == 9, (
                f"Expected action dim 9 (xyz + rot6d) for EEF, got {action.shape[1]}"
            )
            return convert_to_absolute_action_eef(action, reference_state)
        
        elif action_type == ActionType.NON_EEF:
            return convert_to_absolute_action_joints(action, reference_state)
        
        else:
            raise ValueError(f"Unknown ActionType: {action_type}")