# Isaac Deploy IGG Generation Guide

This guide explains how to generate IGG (Isaac Graph Generator) artifacts from GR00T models for deployment with Isaac Deploy. The process exports optimized ONNX models and generates configuration files compatible with Isaac Deploy's inference pipeline.

## 1. Setup Steps

### Prerequisites
Follow the complete setup instructions in the main [README.md](README.md) file in the root directory, including:

- Ubuntu 20.04/22.04 with CUDA 12.4
- Python 3.10 environment
- All dependencies installed via `pip install -e .[base]`
- Flash attention: `pip install --no-build-isolation flash-attn==2.7.1.post4`

### Additional Requirements for IGG Generation

After completing the main setup, you'll need to install the LEAPP library for generating deployment configurations:

```bash
# Install the LEAPP library from the dist folder
pip install dist/leapp-0.1.0-py3-none-any.whl
```

This library provides the `@annotate.method()` decorators and configuration generation capabilities required for IGG artifact creation.

### ONNX Runtime GPU Support (Optional)

For optimal performance during export and validation, install GPU-enabled ONNX Runtime:

```bash
# Uninstall CPU-only version if present
pip uninstall onnxruntime -y

# Install GPU version
pip install onnxruntime-gpu
```

## 2. Export Examples

The repository provides two complete examples in the `export_examples/` directory that demonstrate the end-to-end workflow for generating IGG artifacts.

### 2.1 Handwave Model Example (`isaac_os_handwave_model/`)

This example demonstrates exporting a specialized handwave model:

**Location**: `export_examples/isaac_os_handwave_model/run_handwave_gr00t.py`

**Key Features**:
- Uses `nvidia/GR00T-N1.5-3B-WaveHand-Dev` model
- Configured for Unitree G1 embodiment (`unitree_g1_v2`)
- Focuses on upper body and hand actions
- Generates handwave-specific artifacts

**Usage**:
```bash
cd export_examples/isaac_os_handwave_model
python run_handwave_gr00t.py
```

### 2.2 Vanilla GR00T Example (`vanilla_gr00t/`)

This example shows the standard GR00T model export process:

**Location**: `export_examples/vanilla_gr00t/export_vanilla_gr00t.py`

**Key Features**:
- Uses base `nvidia/GR00T-N1.5-3B` model
- Supports command-line arguments for customization
- Exports full arm and hand control actions
- Uses default demonstration data

**Usage**:
```bash
cd export_examples/vanilla_gr00t
python export_vanilla_gr00t.py --dataset_path demo_data/robot_sim.PickNPlace --model_path nvidia/GR00T-N1.5-3B --save_model_path gr00t_models
```

### 2.3 General Workflow Steps

Both examples follow the same core workflow:

1. **Model and Dataset Loading**
   - Load the GR00T policy with specified embodiment configuration
   - Load demonstration dataset in LeRobot format
   - Configure modality mappings and transforms

2. **Model Export**
   - Export optimized ONNX models for each pipeline component
   - Generate TorchScript models for preprocessing/postprocessing
   - Create deployment-ready model artifacts

3. **Validation and Profiling**
   - Compare original vs. exported model outputs
   - Generate action distribution plots for verification
   - Profile inference performance

4. **IGG Configuration Generation**
   - Generate Isaac Deploy configuration files
   - Create execution graphs with timing information
   - Export backend-specific parameters

## 3. Generated Artifacts

The export process creates a comprehensive set of artifacts organized in the specified output directory (e.g., `gr00t_models/` or `handwave_model/`).

### 3.1 Directory Structure

```
{output_directory}/
├── preprocess/                 # Input preprocessing models
│   ├── preprocess_video.pt     # Video preprocessing (TorchScript)
│   ├── preprocess_state_action.pt  # State preprocessing (TorchScript)
│   └── eagle2_tokenizer.pt     # Text tokenization (TorchScript)
├── eagle2/                     # Vision-Language backbone models
│   ├── vit.onnx               # Vision Transformer (ONNX)
│   └── llm.onnx               # Language Model (ONNX)
├── action_head/               # Action generation models
│   └── denoising_subgraph.onnx  # Diffusion denoising (ONNX)
├── postprocess/               # Output postprocessing
│   └── postprocess_modules.pt  # Action postprocessing (TorchScript)
└── plots/                     # Validation and verification plots
    ├── python/                # Original model outputs
    └── onnx/                  # Exported model outputs
```

### 3.2 Model Components

**Preprocessing Models** (TorchScript):
- `preprocess_video.pt`: Handles video frame normalization and resizing
- `preprocess_state_action.pt`: Processes robot state inputs and action history
- `eagle2_tokenizer.pt`: Tokenizes text instructions for the language model

**Core Models** (ONNX):
- `vit.onnx`: Vision Transformer for image understanding
- `llm.onnx`: Large Language Model for instruction processing and reasoning
- `denoising_subgraph.onnx`: Diffusion-based action generation head

**Postprocessing Models** (TorchScript):
- `postprocess_modules.pt`: Converts raw model outputs to executable robot actions

### 3.3 IGG Configuration Files

The LEAPP library generates Isaac Deploy configuration files during the export process:

**Key Configuration Outputs**:
- **Execution Graph**: Defines the complete inference pipeline topology
- **Backend Parameters**: ONNX model paths and runtime settings  
- **Timing Information**: Performance profiling data for optimization
- **Input/Output Specifications**: Tensor shapes and data types for each component

### 3.4 Validation Plots

The export process generates comprehensive validation plots in the `plots/` directory:

**Python Model Plots** (`plots/python/`):
- Action distribution histograms from original PyTorch model
- Trajectory visualizations and statistics
- Performance baseline measurements

**ONNX Model Plots** (`plots/onnx/`):
- Action distribution histograms from exported ONNX models
- Comparative analysis with original model outputs
- Deployment model validation results

**Generated Plot Types**:
- **`time_series_plot.png`**: Time series visualization showing all action features over multiple timesteps
- **`heatmap_plot.png`**: Heatmap visualization of action features across time for pattern analysis
- **`feature_statistics.png`**: Statistical analysis plots showing mean ± std and min-max ranges per feature

### 3.5 Model Validation Methods

The export process employs a comprehensive three-tier validation system to ensure exported models maintain behavioral consistency with the original PyTorch implementation.

#### 3.5.1 Per-Model Accuracy Verification (Console Output)

Each exported model component undergoes rigorous accuracy validation that is printed to the console:

**TorchScript Model Validation**:
```
comparing: <tensor_name>
The tensors <name> are exactly equal!
```
Or if not exactly equal:
```
The tensors <name> are not exactly equal. Mean absolute difference: <value>
Maximum absolute difference for <name>: <max_diff>, Min diff is <min_diff>
The tensors <name> are approximately equal (within tolerance).
```

**ONNX Model Validation**:
```
Validating ONNX model accuracy...
Accuracy Comparison:
  Max difference: <value>e-<precision>
  Mean difference: <value>e-<precision>
  Within tolerance: True/False
```

**Validation Criteria**:
- **Exact Match**: Preferred outcome where `torch.all(output_export == output_gr00t)` returns True
- **Tolerance Match**: Acceptable when `torch.allclose()` passes with `rtol=1e-2, atol=1e-2`
- **Device Compatibility**: Automatic device matching for cross-device comparisons
- **Shape & Dtype Verification**: Ensures exported models maintain correct tensor specifications

#### 3.5.2 Distribution Consistency Testing (Console Output)

The export process runs models over multiple iterations to verify consistent output distributions:

**Network Consistency Tests**:
```
network consistency test passed over <N> runs
```

**Multi-Iteration Validation** (30 iterations by default):
- Runs the same input through both original and exported models multiple times
- Validates that outputs are deterministic and consistent
- Detects any non-deterministic behavior or export-induced variations
- Console output shows progress: `computing <output_dir> get_actions for iteration <i>`

**Statistical Validation**:
The system calculates and compares:
- Mean absolute differences across iterations
- Maximum deviation from expected outputs  
- Feature-wise consistency metrics
- Distribution shape preservation

#### 3.5.3 Visual Plot Generation and Analysis

**Automated Plot Generation**:
The validation system automatically generates comparative plots by running both PyTorch and ONNX models over 30 iterations:

1. **Action Distribution Analysis**:
   - Collects action outputs from specified joints/actuators (e.g., `action.upper_body`, `action.hands`)
   - Concatenates multi-dimensional action spaces into unified analysis tensors
   - Applies safety bounds (clamps values to [-3.5, 3.5] range)

2. **Generated Visualizations**:
   - **Time Series Plots**: Line graphs showing each action feature over time steps
   - **Heatmap Plots**: Color-coded intensity maps revealing temporal patterns  
   - **Statistical Summary Plots**: Error bars and range indicators for feature-wise analysis

3. **Comparative Validation**:
   - **`plots/python/`**: Baseline results from original PyTorch model
   - **`plots/onnx/`**: Results from exported ONNX models
   - Visual comparison enables quick identification of behavioral drift

**Console Statistics Output**:
```
Time series data: <timesteps> time steps, <features> features
Tensor Statistics:
  Mean: <value>
  Std:  <value>
  Max:  <value>
```

This three-tier validation system ensures that exported models are **deployment-ready** and maintain full behavioral consistency with the original GR00T implementation.

## 4. Usage with Isaac Deploy

Once generated, the artifacts in the output directory can be directly integrated with Isaac Deploy:

1. **Model Integration**: Copy the model directory to Isaac Deploy's model repository
2. **Configuration Loading**: Isaac Deploy automatically loads the IGG configuration files  
3. **Pipeline Execution**: The inference pipeline runs using the optimized ONNX/torchscript models
4. **Performance Monitoring**: Use the generated plots for ongoing validation

The exported models provide significant performance improvements over PyTorch inference while maintaining full compatibility with the original GR00T model behavior.
