# GR00T Model Export

Export GR00T policy models to ONNX format using the leapp framework.

## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git submodule update --init --recursive
```

### 2. Set Up Environment

```bash
uv sync --python 3.10
uv pip install -e .
```

> **Note:** You may need to update the transformers version in `pyproject.toml` to fix input errors:
> ```toml
> "transformers==4.51.3"
> ```

### 3. Validate GR00T Installation

Verify that GR00T works correctly using the test inference script:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --dataset-path demo_data/gr1.PickNPlace \
  --embodiment-tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

### 4. Install leapp

```bash
uv pip install dist/leapp-0.3.0-py3-none-any.whl
```

## Usage

### Export Model

Export the GR00T policy to ONNX format:

```bash
uv run python export/export_with_leapp.py
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | `nvidia/GR00T-N1.6-3B` | Path to the GR00T model |
| `--dataset_path` | str | `demo_data/gr1.PickNPlace` | Path to the dataset for tracing |
| `--embodiment_tag` | str | `gr1` | Embodiment tag for the robot |
| `--video_backend` | str | `torchcodec` | Video decoding backend |
| `--output_name` | str | `exported_gr00t` | Name for the exported model directory |

**Example:**

```bash
uv run python export/export_with_leapp.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --embodiment_tag gr1 \
  --output_name exported_gr00t
```

### Validate Export

Validates the exported model by running both the original and exported policies on the same dataset, then comparing their outputs across all joints.

```bash
uv run python export/policy_comparison.py 
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | `nvidia/GR00T-N1.6-3B` | Path to the GR00T model |
| `--dataset_path` | str | `demo_data/gr1.PickNPlace` | Path to the dataset |
| `--embodiment_tag` | str | `gr1` | Embodiment tag for the robot |
| `--video_backend` | str | `torchcodec` | Video decoding backend |
| `--model_yaml_path` | str | `exported_gr00t/exported_gr00t.yaml` | Path to exported model YAML config |
| `--max_steps` | int | `350` | Number of steps to compare |
| `--use_exported` | bool | `True` | Compare against exported model (onnx) (vs modified model running in pytorch) |
| `--show_plots` | bool | `True` | Display comparison plots interactively |

**Example:**

```bash
uv run python export/policy_comparison.py \
  --model_yaml_path exported_gr00t/exported_gr00t.yaml \
  --max_steps 100 \
  --show_plots True
```


**What it does:**
1. Runs the original PyTorch policy and exported ONNX policy on identical inputs
2. Computes per-dimension error statistics for each joint (left_arm, right_arm, left_hand, right_hand, waist)
3. Generates comparison plots saved as `policy_comparison_<joint>.png`

**Output metrics (per dimension):**
- **Data std**: Standard deviation of the original policy output
- **Mean/Max error**: Absolute difference between original and exported outputs
- **RMSE**: Root mean squared error
- **NRMSE**: Normalized RMSE (RMSE / data_std) — values < 0.1 indicate good agreement
- **Percentiles**: Error distribution (p60, p75, p90, p99, p99.9)

**Generated plots:**
Each plot shows ground truth actions, original policy predictions, and exported policy predictions over time for visual comparison. Plots are saved to the current directory:
- `policy_comparison_left_arm.png`
- `policy_comparison_right_arm.png`
- `policy_comparison_left_hand.png`
- `policy_comparison_right_hand.png`
- `policy_comparison_waist.png`

