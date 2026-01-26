"""
Policy modifications for making GR00T traceable/exportable.

This module contains the modifications needed to convert the GR00T policy
to use PyTorch-traceable operations for ONNX export.
"""

import torch
import types

from data.nn_modifications import get_modified_vision_model, get_modified_language_model
from data.state_action_processor_torch import StateActionProcessorTorch
from data.processor_torch import create_torch_processor
from data.collator_torch import create_torch_collator

from gr00t.policy.gr00t_policy import _rec_to_dtype
from gr00t.data.types import MessageType


def get_action_traceable(self, data, initial_noise=None):
    """
    Torch-traceable version of get_action for export.
    
    This replaces the original get_action method with one that uses
    PyTorch operations instead of PIL/numpy for image processing.
    
    Args:
        data: Input observation data
        initial_noise: Optional initial noise tensor for diffusion.
            Shape: [B, action_horizon, action_dim]. If None, noise is generated internally.
    """
    from leapp import annotate
    
    # Step 1: Split batched observation into individual observations
    unbatched_observations = self._unbatch_observation(data)
    processed_inputs = []
    
    # Convert to torch tensors
    for i in range(len(unbatched_observations)):
        for k, v in unbatched_observations[i]["state"].items():
            unbatched_observations[i]["state"][k] = torch.from_numpy(v)
        for k, v in unbatched_observations[i]["video"].items():
            unbatched_observations[i]["video"][k] = torch.from_numpy(v).to(torch.float32)

    # Annotate inputs for export tracing
    for i in range(len(unbatched_observations)):
        for k, v in unbatched_observations[i]["state"].items():
            unbatched_observations[i]["state"][k] = annotate.input_tensors(
                {k: v}, node_name='preprocess_state'
            )
        for k, v in unbatched_observations[i]["video"].items():
            unbatched_observations[i]["video"][k] = annotate.input_tensors(
                {k: v}, node_name='preprocess_video'
            )

    # Step 2: Process each observation through the VLA processor
    states = []
    for obs in unbatched_observations:
        vla_step_data = self._to_vla_step_data(obs)
        states.append(vla_step_data.states.copy())
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_input = self.processor(messages)

        # Video processing with torch-traceable pipeline
        torch_result = self.torch_processor.process_vlm_inputs_torch(
            vla_step_data, self.embodiment_tag
        )
        processed_input['vlm_content']['images'] = torch_result['images']
        processed_input['vlm_content']['conversation'] = torch_result['conversation']
        processed_inputs.append(processed_input)

    # Step 3: Collate processed inputs into a single batch for model
    collated_inputs = self.collate_fn(processed_inputs)
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.float32)
    
    # Annotate outputs for export
    static_outputs = {
        'input_ids': collated_inputs['inputs']['input_ids'],
        'attention_mask': collated_inputs['inputs']['attention_mask'],
        'image_sizes': collated_inputs['inputs']['image_sizes'],
        'embodiment_id': collated_inputs['inputs']['embodiment_id']
    }
    annotate.output_tensors(
        'preprocess_video',
        {'pixel_values': collated_inputs['inputs']['pixel_values']},
        static_outputs=static_outputs,
        export_with='onnx'
    )
    annotate.output_tensors(
        'preprocess_state',
        {'state': collated_inputs['inputs']['state'],
        'reference': states},
        export_with='onnx', backend_params={'dynamo': False}
    )

    # Step 4: Run model inference to predict actions
    # Use provided initial_noise, or generate one (baked in as constant during export)
    if initial_noise is None:
        batch_size = 1
        action_horizon = self.model.action_head.config.action_horizon
        action_dim = self.model.action_head.action_dim
        device = collated_inputs['inputs']['state'].device
        dtype = collated_inputs['inputs']['state'].dtype
        
        initial_noise = torch.randn(
            batch_size, action_horizon, action_dim,
            device=device, dtype=dtype
        )
    
    with torch.inference_mode():
        model_pred = self.model.get_action(**collated_inputs, initial_noise=initial_noise)
    normalized_action = model_pred["action_pred"].float()

    normalized_action, states = annotate.input_tensors(
        {'normalized_action': normalized_action, 'state': states},
        node_name='decode_action'
    )

    # Step 5: Decode actions from normalized space back to physical units
    batched_states = {}
    for k in self.modality_configs["state"].modality_keys:
        batched_states[k] = torch.stack(
            [s[k] for s in states], dim=0
        ).to(normalized_action.device)
    
    unnormalized_action = self.processor.decode_action(
        normalized_action, self.embodiment_tag, batched_states
    )

    # Cast all actions to float32 for consistency
    casted_action = {
        key: value.to(torch.float32) for key, value in unnormalized_action.items()
    }
    annotate.output_tensors('decode_action', casted_action, export_with='onnx')

    return casted_action, {}


def make_modifications(policy):
    """
    Apply all modifications to make the policy torch-traceable for export.
    
    This modifies the policy in-place to:
    1. Replace vision/language models with export-compatible versions
    2. Replace state/action processor with torch-traceable version
    3. Replace collator with torch-traceable version
    4. Replace get_action method with traceable version
    
    Args:
        policy: Gr00tPolicy instance to modify
        
    Returns:
        Modified policy
    """
    # ==================== Backbone modifications ====================
    vision_model_module = get_modified_vision_model(
        policy.model.backbone.model.vision_model
    )
    language_model_module = get_modified_language_model(
        policy.model.backbone.model.language_model
    )
    policy.model.backbone.model.vision_model = vision_model_module
    policy.model.backbone.model.language_model = language_model_module
    policy.model.backbone.model.eval()

    # Use float32 (half precision causes significant errors)
    policy.model.backbone = policy.model.backbone.float()
    
    # Disable gradients for export
    policy.model.backbone.requires_grad_(False)
    
    # ==================== Action head modifications ====================
    policy.model.action_head = policy.model.action_head.float()

    # ==================== Preprocessing modifications ====================
    # Replace state/action processor with torch-traceable version
    policy.processor.state_action_processor = StateActionProcessorTorch(
        policy.processor.state_action_processor
    )
    
    # Create torch processor for VLM input processing
    policy.torch_processor = create_torch_processor(policy.processor)

    # ==================== Collator modifications ====================
    policy.collate_fn = create_torch_collator(
        model_name=policy.model.config.model_name,
        model_type=policy.model.config.backbone_model_type,
        transformers_loading_kwargs={"trust_remote_code": True},
    )

    # ==================== Replace get_action method ====================
    policy.get_action = types.MethodType(get_action_traceable, policy)

    return policy

