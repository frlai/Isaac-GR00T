import os
import gr00t.model.modules.eagle_backbone as eagle_backbone_module
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import importlib
# Import from the local module (using importlib since directory has hyphens)
_siglip2_module = importlib.import_module(
    "gr00t.model.modules.nvidia.Eagle-Block2A-2B-v2.modeling_siglip2")
Siglip2VisionModel = _siglip2_module.Siglip2VisionModel

def get_modified_vision_model(original_vision_model):
    eagle_path = os.path.join(
    os.path.dirname(eagle_backbone_module.__file__), "nvidia", "Eagle-Block2A-2B-v2")
    config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
    config.vision_config._attn_implementation = 'eager'

    vision_model = Siglip2VisionModel(config.vision_config)
    # Copy weights from the policy's trained model
    vision_model.load_state_dict(original_vision_model.state_dict())
    original_dtype = next(original_vision_model.parameters()).dtype
    vision_model.eval().to(original_dtype).cuda()
    return vision_model

def get_modified_language_model(original_language_model):
    eagle_path = os.path.join(
    os.path.dirname(eagle_backbone_module.__file__), "nvidia", "Eagle-Block2A-2B-v2")
    config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
    # Match the number of layers in the original (truncated) model
    original_num_layers = len(original_language_model.model.layers)
    config.text_config.num_hidden_layers = original_num_layers
    # and often compatible with ONNX export
    config.text_config._attn_implementation = "eager"

    # test for support expantion
    assert config.text_config.architectures[0] == "Qwen3ForCausalLM"
    language_model = Qwen3ForCausalLM(config.text_config)
    language_model.load_state_dict(original_language_model.state_dict())
    original_dtype = next(original_language_model.parameters()).dtype
    language_model.eval().to(original_dtype).cuda()
    return language_model