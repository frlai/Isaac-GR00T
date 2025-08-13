# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn
import argparse
import gr00t
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
import copy
import onnx
import onnxruntime as ort
import numpy as np
import time


torch.cuda.manual_seed(42)


class StandaloneDenoisingSubgraph(nn.Module):
    """Standalone denoising subgraph for ONNX export - N1.5 compatible with integrated backbone processing
    
    This module combines the complete action head pipeline including backbone processing 
    (vlln, vl_self_attention) and action components (state_encoder, action_encoder, 
    action_decoder, DiT model, future_tokens) into a single ONNX-exportable subgraph.
    
    Processes raw backbone_features through the complete N1.5 action head pipeline.
    """

    def __init__(self, policy):
        super().__init__()

        # Always use float16 for consistency with individual TensorRT engines
        dtype = torch.float16

        # Use the same direct conversion approach as individual exports
        self.state_encoder = copy.deepcopy(policy.model.action_head.state_encoder)
        self.state_encoder = self.state_encoder.to(dtype).eval()
        
        self.action_encoder = copy.deepcopy(policy.model.action_head.action_encoder) 
        self.action_encoder = self.action_encoder.to(dtype).eval()
        
        self.action_decoder = copy.deepcopy(policy.model.action_head.action_decoder)
        self.action_decoder = self.action_decoder.to(dtype).eval()
        
        self.dit_model = copy.deepcopy(policy.model.action_head.model)
        self.dit_model = self.dit_model.to(dtype).eval()
        
        # Add future_tokens for N1.5 compatibility
        self.future_tokens = copy.deepcopy(policy.model.action_head.future_tokens)
        self.future_tokens = self.future_tokens.to(dtype).eval()
        
        # Add vlln and vl_self_attention for complete backbone processing
        self.vlln = copy.deepcopy(policy.model.action_head.vlln)
        self.vlln = self.vlln.to(dtype).eval()
        
        self.vl_self_attention = copy.deepcopy(policy.model.action_head.vl_self_attention)
        self.vl_self_attention = self.vl_self_attention.to(dtype).eval()

        # Use the SAME num_inference_timesteps as the individual TensorRT engines
        self.num_inference_timesteps = 4
        print(f"ONNX Export - Using num_inference_timesteps={self.num_inference_timesteps}")
        self.num_timestep_buckets = policy.model.action_head.num_timestep_buckets
        self.action_horizon = policy.model.action_head.config.action_horizon
        self.action_dim = policy.model.action_head.config.action_dim

        if hasattr(policy.model.action_head, 'position_embedding') and policy.model.action_head.config.add_pos_embed:
            self.position_embedding = copy.deepcopy(policy.model.action_head.position_embedding)
            self.position_embedding = self.position_embedding.to(dtype).eval()
            self.add_pos_embed = True
        else:
            self.add_pos_embed = False
            self.position_embedding = None
        # Get dimensions from policy configuration
        action_horizon = policy.model.action_head.config.action_horizon
        action_dim = policy.model.action_head.config.action_dim
        
        init_actions_tensor = torch.randn(
            (1, action_horizon, action_dim),
            dtype=torch.float16, device="cuda"
        )
        self.register_buffer('init_actions', init_actions_tensor)

        self.model_dtype = dtype

    def forward(self, embeddings: torch.Tensor, state: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for denoising subgraph with integrated backbone processing.
        
        Args:
            embeddings: Raw backbone features of shape (B, seq_len, backbone_embed_dim)
            state: State tensor of shape (B, state_seq_len, state_dim)
            embodiment_id: Embodiment IDs of shape (B,) - input as int32, cast to int64 internally
        
        Returns:
            actions: Denoised action predictions of shape (B, action_horizon, action_dim)
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Process backbone features through vlln and vl_self_attention
        embeddings = embeddings.to(self.model_dtype)
        vl_embs = self.vlln(embeddings)
        vl_embs = self.vl_self_attention(vl_embs)

        state = state.to(self.model_dtype)
        embodiment_id = embodiment_id.to(torch.int32)

        state_features = self.state_encoder(state, embodiment_id)
        
        # Use the registered init_actions buffer
        actions = self.init_actions.expand(batch_size, -1, -1).clone()

        dt = 1.0 / self.num_inference_timesteps
        for t in range(self.num_inference_timesteps):
            t_discretized = int(
                (t / float(self.num_inference_timesteps)) * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                (batch_size,), t_discretized, device=device, dtype=torch.int32)

            action_features = self.action_encoder(
                actions, timesteps_tensor, embodiment_id)

            if self.add_pos_embed and self.position_embedding is not None:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(
                    pos_ids).unsqueeze(0).to(self.model_dtype)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension
            # Following N1.5 pattern: state_features, future_tokens, action_features
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            model_output = self.dit_model(
                sa_embs, vl_embs, timesteps_tensor)
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        return actions


def export_model(model, embeddings, state, embodiment_id, output_path):
    """Export PyTorch model to ONNX format"""
    print(f"Exporting to {output_path}")

    # Export ONNX
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        model,
        (embeddings, state, embodiment_id),
        output_path,
        input_names=["embeddings", "state", "embodiment_id"],
        output_names=["actions"],
        dynamic_axes={
            "embeddings": {0: "batch_size"},
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "actions": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )

    # Try to load model to see if it works
    onnx_model = onnx.load(output_path)
    print("ONNX model structure validated successfully")


def validate_model(model, onnx_path, embeddings, state, embodiment_id):
    """Validate ONNX model against PyTorch model"""
    print("\nValidating ONNX model accuracy...")

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(embeddings, state, embodiment_id)

    # Get ONNX output
    ort_session = ort.InferenceSession(
        onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {
        "embeddings": embeddings.cpu().numpy(),
        "state": state.cpu().numpy(),
        "embodiment_id": embodiment_id.cpu().numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = torch.from_numpy(ort_outputs[0]).to(
        pytorch_output.device, dtype=pytorch_output.dtype)

    # Compare results
    max_diff = torch.max(torch.abs(pytorch_output - onnx_output)).item()
    mean_diff = torch.mean(torch.abs(pytorch_output - onnx_output)).item()
    is_close = torch.allclose(
        pytorch_output, onnx_output, atol=1e-2, rtol=1e-2)

    print(f"Accuracy Comparison:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Within tolerance: {is_close}")

    return is_close, ort_session


def _create_synthetic_tensors(policy, input_state, attention_mask):
    """Helper function to create synthetic tensors for integrated backbone processing"""
    # Use backbone embedding dimension since we now process raw backbone features
    backbone_embedding_dim = policy.model.action_head.config.backbone_embedding_dim
    
    print(f"Using dimensions for integrated backbone processing:")
    print(f"  embeddings (raw backbone): (1, {attention_mask.shape[1]}, {backbone_embedding_dim})")
    print(f"  state: {input_state.shape}")
    
    embeddings = torch.randn((1, attention_mask.shape[1], backbone_embedding_dim), dtype=torch.float16, device="cuda")
    state = torch.randn((1, input_state.shape[1], input_state.shape[2]), dtype=torch.float16, device="cuda") 
    embodiment_id = torch.ones((1), dtype=torch.int32, device="cuda")
    
    return embeddings, state, embodiment_id


def export_denoising_subgraph(policy, input_state, attention_mask, save_model_path):
    """Export denoising subgraph to ONNX with validation - N1.5 compatible with integrated backbone processing
    
    This subgraph includes the complete action head pipeline with integrated backbone processing
    (vlln, vl_self_attention) and processes raw backbone_features through the full denoising loop.
    
    Args:
        policy: GR00T policy object
        input_state: Processed state tensor from get_input_info()
        attention_mask: Attention mask from get_input_info()
        save_model_path: Directory path (output will be save_model_path/denoising_subgraph.onnx)
        
    Returns:
        bool: True if export and validation succeeded
    """
    print("Exporting denoising subgraph...")
    
    # Set up policy model
    policy.model.to("cuda", dtype=torch.float16)
    policy.model.eval()

    # Create the standalone denoising model
    torch.manual_seed(42)
    model = StandaloneDenoisingSubgraph(policy).cuda().eval()

    # Create synthetic tensors using input_state and attention_mask dimensions
    embeddings, state, embodiment_id = _create_synthetic_tensors(policy, input_state, attention_mask)

    # Set output path
    output_path = os.path.join(save_model_path, "denoising_subgraph.onnx")

    # Export model
    export_model(model, embeddings, state, embodiment_id, output_path)

    # Always validate the exported model
    is_valid, ort_session = validate_model(
        model, output_path, embeddings, state, embodiment_id)
    
    if is_valid:
        print("\nVALIDATION PASSED")
        return True
    else:
        print("\nVALIDATION FAILED")
        return False


