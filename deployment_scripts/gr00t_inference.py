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

import argparse
import os
from functools import partial

import numpy as np
import torch
from action_head_utils import action_head_pytorch_forward
from trt_model_forward import setup_tensorrt_engines, setup_denoising_subgraph_engine

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


class SyntheticDataGenerator:
    """Generates synthetic input data for GR00T inference testing."""

    STATE_DIMS = {
        'left_arm': 7,
        'left_hand': 6,
        'right_arm': 7,
        'right_hand': 6,
    }
    VIDEO_DIMS = (1, 1, 256, 256, 3)
    POSITION_IDS_LENGTH = 256
    RANDOM_SEED = 42

    def __init__(self, seed=42):
        """Initialize the synthetic data generator."""
        self.seed = seed
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self._create_synthetic_data()
        self._create_position_ids()

    def _create_synthetic_data(self) -> None:
        """Create synthetic data pool similar to reference implementation."""
        self.synthetic_data = {
            'state_left_arm': self._generate_state_data(self.STATE_DIMS['left_arm']),
            'state_left_hand': self._generate_state_data(self.STATE_DIMS['left_hand']),
            'state_right_arm': self._generate_state_data(self.STATE_DIMS['right_arm']),
            'state_right_hand': self._generate_state_data(self.STATE_DIMS['right_hand']),
            'video_data': self._generate_video_data(),
            'task_description': 'pick the pear from the counter and place it in the plate',
        }

    def _generate_state_data(self, dim: int) -> np.ndarray:
        """Generate synthetic state data."""
        return (self.rng.random((1, 1, dim)) - 0.5).astype(np.float32)

    def _generate_video_data(self) -> np.ndarray:
        """Generate synthetic video data."""
        return (self.rng.random(self.VIDEO_DIMS) * 255).astype(np.uint8)

    def _create_position_ids(self) -> None:
        """Create fixed position IDs."""
        try:
            import cupy as cp
            self.fixed_position_ids = cp.arange(0, self.POSITION_IDS_LENGTH, dtype=cp.int32).reshape(
                1, self.POSITION_IDS_LENGTH
            )
        except ImportError:
            # Fallback to numpy if CuPy is not available
            self.fixed_position_ids = np.arange(0, self.POSITION_IDS_LENGTH, dtype=np.int32).reshape(
                1, self.POSITION_IDS_LENGTH
            )

    def create_step_data(self, dataset):
        """
        Create synthetic step data that matches the dataset structure.

        Args:
            dataset: The dataset to get structure from

        Returns:
            Synthetic step data for inference
        """
        # Get a sample from the dataset to understand the structure
        sample_data = dataset[0]

        # Create synthetic data using the pre-generated pool
        synthetic_step_data = {}

        for key, value in sample_data.items():
            # Map to pre-generated synthetic data
            state_key = self._map_state_key(key)
            if state_key and state_key in self.synthetic_data:
                synthetic_step_data[key] = self.synthetic_data[state_key]
            elif 'video' in key.lower() or 'image' in key.lower():
                synthetic_step_data[key] = self.synthetic_data['video_data']
            elif key == 'task_description':
                synthetic_step_data[key] = self.synthetic_data['task_description']
            elif 'position' in key.lower() and hasattr(self, 'fixed_position_ids'):
                synthetic_step_data[key] = self.fixed_position_ids
            else:
                # Keep original data for unmapped keys
                synthetic_step_data[key] = value

        return synthetic_step_data

    def _map_state_key(self, key: str) -> str:
        """Map dataset state key to synthetic data key."""
        key_lower = key.lower()
        if 'left_arm' in key_lower:
            return 'state_left_arm'
        elif 'left_hand' in key_lower:
            return 'state_left_hand'
        elif 'right_arm' in key_lower:
            return 'state_right_arm'
        elif 'right_hand' in key_lower:
            return 'state_right_hand'
        return None


def create_synthetic_input_data(dataset, seed=42):
    """
    Create synthetic input data for testing using structured data pool.

    Args:
        dataset: The dataset to get structure from
        seed: Random seed for reproducibility

    Returns:
        Synthetic step data for inference
    """
    generator = SyntheticDataGenerator(seed)
    return generator.create_step_data(dataset)


def log_prediction_stats(predictions, mode_name, prediction_count=1):
    """
    Log detailed statistics for predictions

    Args:
        predictions: Prediction results dictionary
        mode_name: Name of the inference mode (PyTorch/TensorRT)
        prediction_count: Current prediction number
    """
    print(f"\n=== {mode_name} Inference Output #{prediction_count} ===")

    for tensor_name, tensor_data in predictions.items():
        if tensor_data is not None:
            try:
                # Handle different tensor types
                if hasattr(tensor_data, 'cpu'):
                    tensor_array = tensor_data.cpu().numpy()
                elif hasattr(tensor_data, '__dlpack__'):
                    # Handle CuPy tensors or other DLPack compatible tensors
                    try:
                        import cupy as cp
                        tensor_array = cp.from_dlpack(
                            tensor_data.__dlpack__()).get()
                    except ImportError:
                        tensor_array = tensor_data
                else:
                    tensor_array = tensor_data

                if tensor_array.size > 0:
                    print(
                        f"  {tensor_name}: {tensor_array.shape} {tensor_array.dtype} "
                        f"[min: {tensor_array.min():.3f}, max: {tensor_array.max():.3f}, "
                        f"mean: {tensor_array.mean():.3f}]"
                    )


<< << << < HEAD
                    print(f"    Values: {tensor_array.flatten()[:5]}")
== == == =
>>>>>> > frlai/export_tokenizer_improc
                else:
                    print(
                        f"  {tensor_name}: {tensor_array.shape} {tensor_array.dtype} [empty]")
            except (RuntimeError, ValueError, AttributeError) as e:
                print(f"  {tensor_name}: <error processing tensor: {e}>")
        else:
            print(f"  {tensor_name}: None")


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, 'compare' for compare PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Use synthetic data instead of dataset data for more controlled testing",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    EMBODIMENT_TAG = "gr1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    # Choose input data source
    if args.use_synthetic_data:
        print("Using synthetic input data for controlled testing")
        step_data = create_synthetic_input_data(dataset)
    else:
        print("Using dataset sample for input")
        step_data = dataset[0]
    
    # Log input data statistics
    print("\n=== Input Data Statistics ===")
    for key, value in step_data.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if hasattr(value, 'shape'):
                if args.use_synthetic_data and ('state' in key.lower() or 'position' in key.lower()):
                    # Print actual tensor values for state data and position IDs when using synthetic data
                    print(f"  {key}: {value.shape} {value.dtype}")
                    if 'position' in key.lower():
                        print(f"    Values: {value.flatten()}")
                    else:
                        print(f"    Mean: {value.mean():.6f}")
                        print(f"    Min: {value.min():.6f}")
                        print(f"    Max: {value.max():.6f}")
                elif value.size > 0:
                    print(f"  {key}: {value.shape} {value.dtype} [min: {value.min():.3f}, max: {value.max():.3f}, mean: {value.mean():.3f}]")
                else:
                    print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    prediction_count = 1

    if args.inference_mode == "pytorch":
        torch.cuda.manual_seed(42)
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
            print(f"\n=== Generated init_actions ===")
            print(f"  Shape: {policy.model.action_head.init_actions.shape}")
            print(f"  Mean: {policy.model.action_head.init_actions.mean():.6f}")

        print(f"  num_inference_timesteps: {policy.model.action_head.num_inference_timesteps}")
        predicted_action = policy.get_action(step_data)
        log_prediction_stats(predicted_action, "PyTorch", prediction_count)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        torch.cuda.manual_seed(42)
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        setup_denoising_subgraph_engine(policy, args.trt_engine_path)

        predicted_action = policy.get_action(step_data)
        log_prediction_stats(predicted_action, "TensorRT", prediction_count)

    else:
        # ensure PyTorch and TensorRT have the same init_actions
        torch.cuda.manual_seed(42)
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
            print(f"\n=== Generated init_actions ===")
            print(f"  Shape: {policy.model.action_head.init_actions.shape}")
            print(f"  Mean: {policy.model.action_head.init_actions.mean():.6f}")
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)
        log_prediction_stats(predicted_action_torch, "PyTorch", prediction_count)

        # Setup TensorRT engines and run inference
        setup_denoising_subgraph_engine(policy, args.trt_engine_path)
        predicted_action_tensorrt = policy.get_action(step_data)
        log_prediction_stats(predicted_action_tensorrt, "TensorRT", prediction_count)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
