# N1.7 Leapp Export Change Summary

This summarizes the current working-tree changes needed to port the previous N1.6 leapp export path to N1.7/Qwen3-VL.

| Change type | Files impacted | Size of change | Description / justification |
| --- | --- | ---: | --- |
| Qwen3-VL model patching | `export/data/nn_modifications.py` | 209 lines changed | Replaced the obsolete N1.6 Eagle/SigLIP replacement logic with N1.7 Qwen3-VL monkey patches. The vision path patches Qwen3-VL attention to avoid complex rotary ops and trace-unfriendly attention handling. The language path patches DeepStack injection and the text model forward to use a simple ONNX-friendly causal mask instead of the Hugging Face masking path that failed during export. |
| Torch-traced Qwen image preprocessing | `export/data/collator_torch.py` | 266 lines changed | Reworked the old Eagle-style collator into a Qwen3-VL compatible shim. It now produces `pixel_values` and `image_grid_thw`, and patchifies images using torch operations so leapp tracing is preserved. Dummy PIL images are used only for tokenizer placeholder expansion, while the exported image tensor remains traced. |
| Functional tensor construction for decode | `export/data/utils_torch.py` | 26 lines changed | Replaced partial-slice assignments in homogeneous matrix construction/inversion with functional `torch.cat` construction. Leapp warned that partial assignment into a plain tensor breaks trace propagation, which caused `decode_action` outputs to be rejected as non-traced. |
| N1.7 policy hookup | `export/policy_modifications.py` | 35 lines changed | Updated `make_modifications()` to patch the live N1.7 Qwen3-VL modules at `policy.model.backbone.model.model.visual` and `.language_model` instead of removed N1.6 `vision_model`/`language_model` attributes. Also changed the static video output key from `image_sizes` to `image_grid_thw`, and updated the action-head wrapper signature to include N1.7's `options` argument. |
| N1.7 defaults | `export/export_with_leapp.py`, `export/utils.py` | 19 lines changed | Updated export defaults from N1.6 GR1 to N1.7 DROID: `nvidia/GR00T-N1.7-3B`, `demo_data/droid_sample`, and `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`. Also switched embodiment parsing to `EmbodimentTag.resolve()` so N1.7 enum aliases resolve correctly. |
| DROID leapp metadata | `export/joint_name_parser.py` | 53 lines changed | Added DROID joint/element-name metadata for leapp tensor semantics. The DROID representation uses `eef_9d` (9D), `gripper_position` (1D), and `joint_position` (7D), so the registry now provides stable element names with correct dimensions. This affects annotation metadata, not model math. |
| N1.7 processor tensor compatibility | `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | 11 lines changed | Adjusted state concatenation to accept either NumPy arrays or tensor-like values. Leapp annotations turn normalized state arrays into traced tensor subclasses, and the previous unconditional `torch.from_numpy()` failed on those traced tensors. |

Overall diff size at the time of writing:

```text
8 files changed, 393 insertions(+), 226 deletions(-)
```

The export was verified with:

```bash
source .venv/bin/activate
python export/export_with_leapp.py
```

That run completed successfully and wrote the exported graph to `./exported_gr00t`.
