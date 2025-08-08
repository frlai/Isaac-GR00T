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
import time
from typing import Optional
import torch
from gr00t.model.backbone.eagle_backbone import EagleBackbone, DEFAULT_EAGLE_MODEL_NAME
from transformers import AutoConfig, LlamaForCausalLM
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipVisionEmbeddings
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from gr00t.model.policy import unsqueeze_dict_values


def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    normalized_input = unsqueeze_dict_values
    # Apply transforms
    normalized_input = policy.apply_transforms(observations)

    return (
        normalized_input["attention_mask"],
        normalized_input["state"],
        normalized_input["pixel_values"],
        normalized_input["input_ids"],
    )


def export_eagle2_vit(vision_model, output_dir):
    class SiglipVisionEmbeddingsOpt(SiglipVisionEmbeddings):
        def __init__(self, config):
            super().__init__(config)

        def forward(
            self,
            pixel_values: torch.FloatTensor,
            # CHANGED: Use IntTensor (int32) instead of LongTensor (int64)
            position_ids: torch.IntTensor,
            interpolate_pos_encoding=False,
        ) -> torch.Tensor:
            _, _, height, width = pixel_values.shape
            patch_embeds = self.patch_embedding(
                pixel_values)  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if interpolate_pos_encoding:
                embeddings = embeddings + \
                    self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

    class SiglipVisionTransformerOpt(SiglipVisionTransformer):
        def __init__(self, config: SiglipVisionConfig):
            config._attn_implementation = "eager"
            super().__init__(config)
            self.embeddings = SiglipVisionEmbeddingsOpt(config)
            self.head = torch.nn.Identity()

        def forward(
            self,
            pixel_values,
            position_ids,  # Pass position_ids as input
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = False,
        ):
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            hidden_states = self.embeddings(
                pixel_values, position_ids=position_ids, interpolate_pos_encoding=interpolate_pos_encoding
            )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state

    model = SiglipVisionTransformerOpt(vision_model.config).to(torch.float16)
    model.load_state_dict(vision_model.state_dict())
    model.eval().cuda()

    pixel_values = torch.randn(
        (1, model.config.num_channels,
         model.config.image_size, model.config.image_size),
        dtype=torch.float16,
        device="cuda",
    )
    # CHANGED: Use int32 instead of int64 for position_ids
    position_ids = torch.arange(
        model.embeddings.num_patches, device="cuda", dtype=torch.int32).expand((1, -1))
    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, position_ids),  # Include position_ids in ONNX export
            f"{output_dir}/eagle2/vit.onnx",
            # Add position_ids to input names
            input_names=["pixel_values", "position_ids"],
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )


def export_eagle2_llm(backbone_model, backbone_config, output_dir, attention_mask, input_ids):
    class EagleBackboneOpt(EagleBackbone):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Modify LlamamModel architecture for ONNX export
            config = AutoConfig.from_pretrained(
                DEFAULT_EAGLE_MODEL_NAME, trust_remote_code=True)
            config.llm_config._attn_implementation = "eager"

            assert config.llm_config.architectures[0] == "LlamaForCausalLM"
            self.model.language_model = LlamaForCausalLM(config.llm_config)

            # remove parts of the LLM
            self.model.language_model.lm_head = torch.nn.Identity()
            while len(self.model.language_model.model.layers) > kwargs["select_layer"]:
                self.model.language_model.model.layers.pop(-1)

        def forward(self, input_ids, vit_embeds, attention_mask):
            # Post process of vit_embeds according to extract_feature
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], h, w, -1)  # [B, 16, 16, 1152]

            vit_embeds = self.model.pixel_shuffle(
                vit_embeds, scale_factor=self.model.downsample_ratio
            )  # [B, 8, 8, 4608]
            vit_embeds = vit_embeds.reshape(
                # [B, 64, 4608])
                vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.model.mlp1(vit_embeds)  # [1, 64, 2048]

            # Merge input_embeds and vit_embeds
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.model.img_context_token_id

            embeds_to_scatter = vit_embeds.reshape(-1, C).to(
                input_embeds.device, input_embeds.dtype)
            # input_embeds[selected] = embeds_to_scatter
            # Since selected is always a contiguous block [0,0,0,1,1,1,1,0,0,0],
            # we can find start and length without using nonzero (which creates ONNX issues)
            selected_float = selected.float()
            num_vision_tokens = selected.sum().item()

            if num_vision_tokens > 0:
                # Find the start index of the first 1 using argmax
                start_idx = torch.argmax(selected_float).item()

                # Replace the contiguous block with vision embeddings using simple slicing
                input_embeds[start_idx: start_idx +
                             num_vision_tokens] = embeds_to_scatter[:num_vision_tokens]

            # print(input_embeds.shape)
            # print(embeds_to_scatter.shape)
            # print(selected.shape)
            # print(sum(selected))

            # LLM forward
            input_embeds = input_embeds.reshape(B, N, C)
            embeddings = self.model.language_model.forward(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            embeddings = embeddings.hidden_states[-1]

            embeddings = self.linear(embeddings)  # [1, 99, 1536]

            return embeddings

    model = EagleBackboneOpt(**backbone_config).to(torch.float16)
    model.load_state_dict(backbone_model.state_dict())
    model.eval().cuda()

    input_ids = input_ids.to(device="cuda", dtype=torch.int32)
    # Count how many vision tokens are needed by counting img_context_token_id in input_ids
    num_vision_tokens = (
        input_ids == backbone_model.model.img_context_token_id).sum().item()
    # Calculate batch size needed for vit_embeds
    vision_batch_size = num_vision_tokens // model.model.vision_model.vision_model.embeddings.num_patches
    if vision_batch_size == 0:
        vision_batch_size = 1

    vit_embeds = torch.randn(
        (
            vision_batch_size,
            model.model.vision_model.vision_model.embeddings.num_patches,
            model.model.vision_model.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    attention_mask = torch.ones(
        (1, attention_mask.shape[1]), dtype=torch.int32).cuda()

    # print(model(input_ids, vit_embeds, attention_mask).shape)
    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_ids, vit_embeds, attention_mask),
            f"{output_dir}/eagle2/llm.onnx",
            input_names=["input_ids", "vit_embeds", "attention_mask"],
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            # Remove dynamic_axes to export with fixed shapes - eliminates need for set_runtime_tensor_shape
            # dynamic_axes={
            #     "input_ids": {0: "batch_size"},
            #     "vit_embeds": {0: "video_batch_size"},
            #     "attention_mask": {0: "batch_size"},
            #     "embeddings": {0: "batch_size"},  # Only batch dimension is dynamic
            # },
        )


def export_action_head(policy, ONNX_export_path, input_state, attention_mask):
    start_time = time.time()

    # Create temp directory for individual action head components
    temp_action_head_path = os.path.join(
        ONNX_export_path, "action_head", "temp")
    os.makedirs(temp_action_head_path, exist_ok=True)

    state_encoder = policy.model.action_head.state_encoder.to(torch.float16)

    state_tensor = torch.randn(
        (1, input_state.shape[1], input_state.shape[2]), dtype=torch.float16).cuda()
    # CHANGED: Use int32 instead of int64 for embodiment_id
    embodiment_id_tensor = torch.ones((1), dtype=torch.int32).cuda()

    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id_tensor),
        os.path.join(temp_action_head_path, "state_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={"state": {0: "batch_size"}, "embodiment_id": {
            0: "batch_size"}, "output": {0: "batch_size"}},
    )

    action_encoder = policy.model.action_head.action_encoder.to(torch.float16)
    actions_tensor = torch.randn(
        (1, policy.model.action_head.config.action_horizon,
         policy.model.action_head.config.action_dim),
        dtype=torch.float16,
    ).cuda()
    # CHANGED: Use int32 instead of int64 for timesteps_tensor
    timesteps_tensor = torch.ones((1), dtype=torch.int32).cuda()

    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id_tensor),
        os.path.join(temp_action_head_path, "action_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    DiT = policy.model.action_head.model.to(torch.float16)
    sa_embs_tensor = torch.randn(
        (
            1,
            input_state.shape[1] +
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.input_embedding_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    vl_embs_tensor = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.input_embedding_dim), dtype=torch.float16
    ).cuda()

    torch.onnx.export(
        DiT,
        (sa_embs_tensor, vl_embs_tensor, timesteps_tensor),
        os.path.join(temp_action_head_path, "DiT.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            # 'vl_embs': {0: 'batch_size', 1: 'sequence_length'},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_decoder = policy.model.action_head.action_decoder.to(torch.float16)
    model_output_tensor = torch.randn(
        (
            1,
            input_state.shape[1] +
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    torch.onnx.export(
        action_decoder,
        (model_output_tensor, embodiment_id_tensor),
        os.path.join(temp_action_head_path, "action_decoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
