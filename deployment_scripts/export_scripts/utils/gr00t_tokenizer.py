import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from PIL import Image
import os
from transformers import AutoProcessor


DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

EMBODIMENT_TAG_MAPPING = {'new_embodiment': 31}


class GR00TTransform(nn.Module):
    """
    Standalone tokenizer/transformer that processes only video and language.
    - Implemented as an nn.Module
    - No dependency on EagleProcessor or gr00t.data.*
    - Loads a processor from a provided model path (trust_remote_code=True)
    """

    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(__file__), "eagle2_hg_model"),
        default_instruction: str = "Perform the default behavior.",
        language_key: Optional[str] = None,
        embodiment_tag: str = 'new_embodiment',
        embodiment_tag_mapping: dict[str, int] = EMBODIMENT_TAG_MAPPING,
        ** _: Any,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        # Set left padding if tokenizer exists and supports it
        if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "padding_side"):
            self.processor.tokenizer.padding_side = "left"

        self.default_instruction = default_instruction
        self.language_key = language_key
        self.embodiment_id = embodiment_tag_mapping[embodiment_tag]

    # ---------- helpers ----------
    def _detect_language_key(self, data: Dict[str, Any]) -> None:
        if self.language_key:
            return
        grouped_keys: Dict[str, List[str]] = {}
        for key in data.keys():
            try:
                modality, _ = key.split(".")
            except Exception:  # noqa: E722
                modality = "language" if "annotation" in key else "others"
            grouped_keys.setdefault(modality, []).append(key)
        if "language" in grouped_keys and len(grouped_keys["language"]) > 0:
            language_keys = grouped_keys["language"]
            if len(language_keys) == 1:
                self.language_key = language_keys[0]

    def _prepare_video_numpy(self, video: Any) -> np.ndarray:
        """
        Accepts video as a NumPy array or a Torch tensor with shapes:
        - [T, V, H, W, C] (single sample)
        - [B, T, V, H, W, C] (batched)
        Returns uint8 numpy array for a single sample in shape [V, T, C, H, W].
        """
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        assert video.ndim in (5, 6), f"Unsupported video ndim: {video.ndim}"
        if video.ndim == 6:
            # Caller should slice per-sample before calling this helper
            raise ValueError(
                "_prepare_video_numpy expects a single sample (5D). Got 6D.")
        # video: [T, V, H, W, C] -> [V, T, C, H, W]
        video = video.astype(np.uint8, copy=False)
        video_vtchw = np.transpose(video, (1, 0, 4, 2, 3))
        return video_vtchw

    def _prepare_language(self, data: Dict[str, Any]) -> str:
        if self.language_key is not None and self.language_key in data:
            raw_language = data[self.language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]
        else:
            raw_language = self.default_instruction
        return raw_language

    def _build_conversation(self, images_vtchw: np.ndarray, language: str) -> List[Dict[str, Any]]:
        """
        Build a conversation structure expected by many VLM processors that implement
        apply_chat_template/process_vision_info via trust_remote_code.
        images_vtchw: [V, T, C, H, W]
        """
        v, t, c, h, w = images_vtchw.shape
        # Flatten into frames: [(C,H,W)] with order (t, v)
        # Create PIL images in HWC order per frame
        frames_chw = images_vtchw.transpose(
            1, 0, 2, 3, 4).reshape(t * v, c, h, w)
        frames_hwc = [np.transpose(fr, (1, 2, 0)) for fr in frames_chw]
        pil_images = [Image.fromarray(fr) for fr in frames_hwc]

        text_content = [{"type": "text", "text": language}]
        image_content = [{"type": "image", "image": img} for img in pil_images]
        conversation = [
            {
                "role": "user",
                "content": image_content + text_content,
            }
        ]
        return conversation

    # ---------- core processing ----------
    def _tokenize_with_processor(self, conversations: List[List[Dict[str, Any]]], device: torch.device) -> Dict[str, torch.Tensor]:
        # Build text list using chat templates
        text_list: List[str] = []
        image_inputs_flat: List[Any] = []
        for conv in conversations:
            # Some processors implement apply_chat_template/process_vision_info via remote code
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: simple user message
                text = DEFAULT_SYSTEM_MESSAGE + "\n" + "; ".join(
                    c.get("text", "") if isinstance(c, dict) else "" for c in conv[0].get("content", [])
                )
            text_list.append(text)

            if hasattr(self.processor, "process_vision_info"):
                image_inputs, _ = self.processor.process_vision_info(conv)
            else:
                # Fallback: extract PIL images directly
                pil_images = [c["image"] for c in conv[0]["content"]
                              if isinstance(c, dict) and c.get("type") == "image"]
                image_inputs = pil_images
            # Flatten the image inputs - processor expects a flat list
            if image_inputs is not None:
                if isinstance(image_inputs, list):
                    image_inputs_flat.extend(image_inputs)
                else:
                    image_inputs_flat.append(image_inputs)

        inputs = self.processor(
            text=text_list,
            images=image_inputs_flat if len(image_inputs_flat) > 0 else None,
            return_tensors="pt",
            padding=True,
        )
        # Move to device and prefix keys with 'eagle_'
        outputs: Dict[str, Any] = {}
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(device)
            outputs[f"eagle_{k}"] = v
        return outputs

    def _process_single(self, data: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        self._detect_language_key(data)
        video = data["video"]
        if isinstance(video, torch.Tensor) and video.ndim == 4:
            # Unsqueeze to [T, V, H, W, C] if necessary
            while video.ndim < 5:
                video = video.unsqueeze(0)
        video_np = self._prepare_video_numpy(video)
        language = self._prepare_language(data)
        conversation = self._build_conversation(video_np, language)
        return self._tokenize_with_processor([conversation], device)

    def _process_batch(self, data: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        # Split along batch dimension and build conversations
        conversations: List[List[Dict[str, Any]]] = []
        for i in range(batch_size):
            single: Dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, str):
                    single[key] = value
                else:
                    try:
                        single[key] = value[i]
                    except (TypeError, IndexError):
                        single[key] = value
            self._detect_language_key(single)
            video_np = self._prepare_video_numpy(single["video"])
            language = self._prepare_language(single)
            conversations.append(self._build_conversation(video_np, language))

        return self._tokenize_with_processor(conversations, device)

    def _check_batch(self, data: Dict[str, Any]) -> tuple[bool, int]:
        video = data["video"]
        if isinstance(video, torch.Tensor):
            ndim = video.ndim
        else:
            ndim = np.asarray(video).ndim
        if ndim == 5:  # [T, V, H, W, C]
            return False, 1
        if ndim == 6:  # [B, T, V, H, W, C]
            return True, int(video.shape[0])
        raise ValueError(f"Unsupported video number of dimensions: {ndim}")

    # ---------- nn.Module API ----------
    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Determine device from incoming tensors
        tensor_devices = [
            v.device for v in data.values() if isinstance(v, torch.Tensor)]
        device = tensor_devices[0] if len(
            tensor_devices) > 0 else torch.device("cpu")

        # Normalize video dims to at least 5D for single sample usage in helpers
        if "video" in data and isinstance(data["video"], torch.Tensor) and data["video"].ndim == 4:
            while data["video"].ndim < 6:
                data["video"] = data["video"].unsqueeze(0)

        is_batched, batch_size = self._check_batch(data)
        if is_batched:
            outputs = self._process_batch(data, batch_size, device)
        else:
            outputs = self._process_single(data, device)

        # Optional dtype normalization similar to example
        for k, v in list(outputs.items()):
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bfloat16:
                    outputs[k] = v.to(torch.float16)
                elif v.dtype == torch.int64:
                    outputs[k] = v.to(torch.int32)

        # Always include embodiment_id as int32 tensor on the target device with shape [1]
        outputs["embodiment_id"] = torch.tensor(
            [self.embodiment_id], device=device, dtype=torch.int32
        )

        return outputs
