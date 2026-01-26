"""
Policy comparison utilities for evaluating original vs modified GR00T policy.

This module provides tools to compare the outputs of the original GR00T policy
against the modified (torch-traceable) version to validate export accuracy.
"""

import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from utils import get_policy_and_dataset, get_gr00t_input, set_all_seeds
from policy_modifications import make_modifications
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from leapp.inference_manager import InferenceManager
import os


import argparse
import gr00t

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default='nvidia/GR00T-N1.6-3B')
args.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/gr1.PickNPlace"))
args.add_argument("--embodiment_tag", type=str, default='gr1')
args.add_argument("--video_backend", type=str, default='torchcodec')
args.add_argument("--model_yaml_path", type=str, default='exported_gr00t/exported_gr00t.yaml')
args.add_argument("--max_steps", type=int, default=350)
args.add_argument("--use_exported", type=bool, default=True)
args.add_argument("--show_plots", type=bool, default=True)
args = args.parse_args()


def collect_policy_outputs(policy, dataset, joint_names, max_steps, policy_name, 
                           initial_noise_list=None):
    """
    Run policy on multiple steps and collect outputs as numpy arrays.
    
    Args:
        policy: Gr00tPolicy instance
        dataset: Dataset to sample from
        joint_names: List of joint names to collect
        max_steps: Number of steps to process
        policy_name: Name for logging purposes
        initial_noise_list: Optional list of pre-generated noise tensors (one per step).
            If provided, uses deterministic noise for fair comparison.
        
    Returns:
        Dict mapping joint_name -> numpy array of shape (max_steps, num_dims)
    """
    outputs = {joint_name: [] for joint_name in joint_names}
    
    for step_count in range(max_steps):
        data = get_gr00t_input(dataset, policy, step_index=step_count, step=None)
        
        set_all_seeds(42)
        
        # Use pre-generated noise if provided for deterministic comparison
        if initial_noise_list is not None:
            action, _ = policy.get_action(data, initial_noise=initial_noise_list[step_count])
        else:
            action, _ = policy.get_action(data)
        
        for joint_name in joint_names:
            val = action[joint_name][0, 0]
            val_np = val.cpu().numpy() if torch.is_tensor(val) else val
            outputs[joint_name].append(val_np)
        
        if step_count % 10 == 0:
            print(f"  {policy_name}: Processed step {step_count}/{max_steps}")
    
    # Convert to numpy arrays
    for joint_name in joint_names:
        outputs[joint_name] = np.array(outputs[joint_name])
    
    return outputs


def collect_exported_policy_outputs(exported_policy, dataset, modality_config, 
                                     embodiment_tag, joint_names, video_key, initial_noise_list, max_steps):
    """
    Run exported policy on multiple steps and collect outputs as numpy arrays.
    
    Args:
        exported_policy: InferenceManager instance for the exported model
        episode_data: Episode data from dataset
        modality_config: Modality configuration
        embodiment_tag: Embodiment tag
        joint_names: List of joint names to collect
        video_key: Key for video input (e.g., 'ego_view_bg_crop_pad_res256_freq20')
        max_steps: Number of steps to process
        
    Returns:
        Dict mapping joint_name -> numpy array of shape (max_steps, num_dims)
    """
    outputs = {joint_name: [] for joint_name in joint_names}
    
    for step_count in range(max_steps):
        # Extract step data from dataset
        step_data = extract_step_data(
            dataset[0], step_index=step_count, modality_configs=modality_config,
            embodiment_tag=embodiment_tag, allow_padding=False
        )
        
        # Build inputs dict for exported policy
        # Map joint names to leapp input format: preprocess_state/<joint_name>
        inputs = {}
        for joint_name in joint_names:
            state_key = f"preprocess_state/{joint_name}"
            # States are numpy arrays, convert to tensor
            state_data = step_data.states[joint_name]
            inputs[state_key] = torch.from_numpy(state_data).float()
        
        # Add video input
        # Stack images and convert to tensor - keep as uint8 (model expects raw image data)
        video_data = np.stack(step_data.images[video_key])  # (T, H, W, C)
        inputs[f"preprocess_video/{video_key}"] = torch.from_numpy(video_data).to(torch.float32)  # Keep uint8, don't convert to float
        inputs['action_head/initial_noise'] = initial_noise_list[step_count]
        
        # Run exported policy
        set_all_seeds(42)
        policy_outputs = exported_policy.run_policy(inputs)
        
        # Extract action outputs from decode_action node
        # Output keys are like 'decode_action/<joint_name>'
        for joint_name in joint_names:
            output_key = f"decode_action/{joint_name}"
            if output_key in policy_outputs:
                val = policy_outputs[output_key]
                # Handle tensor or numpy
                if torch.is_tensor(val):
                    val_np = val.cpu().numpy()
                else:
                    val_np = np.array(val)
                # Flatten if needed and take first element
                val_np = val_np.flatten()
                outputs[joint_name].append(val_np)
        
        if step_count % 10 == 0:
            print(f"  Exported: Processed step {step_count}/{max_steps}")
    
    # Convert to numpy arrays
    for joint_name in joint_names:
        if outputs[joint_name]:
            outputs[joint_name] = np.array(outputs[joint_name])
        else:
            outputs[joint_name] = None
    
    return outputs


def collect_ground_truth(episode_data, modality_config, embodiment_tag, joint_names, max_steps):
    """
    Collect ground truth actions from episode data.
    
    Args:
        episode_data: Episode data from dataset
        modality_config: Modality configuration
        embodiment_tag: Embodiment tag
        joint_names: List of joint names
        max_steps: Number of steps to collect
        
    Returns:
        Dict mapping joint_name -> numpy array of shape (max_steps, num_dims)
    """
    gt_outputs = {joint_name: [] for joint_name in joint_names}
    
    for step_count in range(max_steps):
        step_data = extract_step_data(
            episode_data, step_index=step_count, modality_configs=modality_config,
            embodiment_tag=embodiment_tag, allow_padding=False
        )
        for joint_name in joint_names:
            gt_outputs[joint_name].append(step_data.actions[joint_name][0])
    
    for joint_name in joint_names:
        gt_outputs[joint_name] = np.array(gt_outputs[joint_name])
    
    return gt_outputs


def compute_error_statistics(original_data, modified_data):
    """
    Compute error statistics between original and modified outputs.
    
    Args:
        original_data: numpy array of original policy outputs
        modified_data: numpy array of modified policy outputs
        
    Returns:
        Dict with error statistics per dimension
    """
    num_dims = original_data.shape[1]
    stats = []
    
    for dim in range(num_dims):
        orig_dim = original_data[:, dim]
        mod_dim = modified_data[:, dim]
        error_dim = np.abs(orig_dim - mod_dim)
        
        # Data statistics
        data_std = orig_dim.std()
        if data_std == 0:
            data_std = 1.0  # Avoid division by zero
        
        # RMSE and NRMSE
        rmse = np.sqrt(np.mean(error_dim ** 2))
        nrmse = rmse / data_std
        
        # Percentiles
        stats.append({
            'dim': dim,
            'data_std': orig_dim.std(),
            'mean_error': error_dim.mean(),
            'max_error': error_dim.max(),
            'rmse': rmse,
            'nrmse': nrmse,
            'p60': np.percentile(error_dim, 60),
            'p75': np.percentile(error_dim, 75),
            'p90': np.percentile(error_dim, 90),
            'p99': np.percentile(error_dim, 99),
            'p999': np.percentile(error_dim, 99.9),
        })
    
    return stats


def print_error_statistics(joint_name, stats, max_steps):
    """Print formatted error statistics for a joint."""
    num_dims = len(stats)
    
    print(f"\n{'='*90}")
    print(f"{joint_name} ({num_dims} dimensions, {max_steps} steps)")
    print(f"{'='*90}")
    
    for s in stats:
        print(f"\n  Dimension {s['dim']}:")
        print(f"    Data std:      {s['data_std']:.6f}")
        print(f"    Mean error:    {s['mean_error']:.6e}")
        print(f"    Max error:     {s['max_error']:.6e}")
        print(f"    RMSE:          {s['rmse']:.6e}")
        print(f"    NRMSE:         {s['nrmse']:.6e}  (RMSE / data_std)")
        print(f"    Percentiles:   p60={s['p60']:.2e}, p75={s['p75']:.2e}, "
              f"p90={s['p90']:.2e}, p99={s['p99']:.2e}, p99.9={s['p999']:.2e}")


def plot_joint_comparison(joint_name, gt_data, original_data, modified_data, exported, output_dir="."):
    """
    Create comparison plot for a single joint.
    
    Args:
        joint_name: Name of the joint
        gt_data: Ground truth data (num_steps, num_dims)
        original_data: Original policy outputs (num_steps, num_dims)
        modified_data: Modified policy outputs (num_steps, num_dims)
        output_dir: Directory to save plot
    """
    num_dims = gt_data.shape[1]
    
    fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(12, 2.5*num_dims))
    fig.suptitle(f"{joint_name}", fontsize=14, fontweight='bold')
    
    if num_dims == 1:
        axes = [axes]
    
    label = "Exported Policy" if exported else "Modified Policy"
    for i, ax in enumerate(axes):
        ax.plot(gt_data[:, i], label="GT Action", color='green', alpha=0.7, linewidth=2)
        ax.plot(original_data[:, i], label="Original Policy", color='blue', alpha=0.7, linewidth=1.5)
        ax.plot(modified_data[:, i], label=label, color='red', alpha=0.7, linestyle='--', linewidth=1.5)
        
        ax.set_title(f"Dimension {i}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"{output_dir}/policy_comparison_{joint_name}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n  Saved: {output_path}")
    
    return fig


def plot_policy_comparison(max_steps=100, output_dir=".", show_plots=True,
                           use_exported=False, 
                           exported_model_path='test_export_gr00t/test_export_gr00t.yaml',
                           model_path='nvidia/GR00T-N1.6-3B',
                           dataset_path=None,
                           embodiment_tag='gr1',
                           video_backend='torchcodec'):
    """
    Plot comparison of ground truth actions, original GR00T policy outputs, 
    modified policy outputs, and optionally exported policy outputs.
    
    Creates separate figure windows for each joint type.
    Memory-efficient: loads one policy at a time, stores outputs, then frees memory.
    
    Args:
        max_steps: Number of steps to compare
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively
        use_exported: Whether to evaluate against the exported model or the modified model
        exported_model_path: Path to exported model YAML config
        
    Returns:
        Tuple of (gt_outputs, original_outputs, modified_outputs, exported_outputs)
    """
    # Step 1: Get config info (load briefly, then delete)
    print("Loading config info...")
    set_all_seeds(42)
    temp_policy, dataset = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
    modality_config = temp_policy.get_modality_config()
    embodiment_tag = temp_policy.embodiment_tag
    episode_data = dataset[0]
    print(f"Episode length: {len(episode_data)}")
    
    # Get joint names and video key from first step
    first_step_data = extract_step_data(
        episode_data, step_index=0, modality_configs=modality_config,
        embodiment_tag=embodiment_tag, allow_padding=False
    )
    joint_names = list(first_step_data.actions.keys())
    video_keys = list(first_step_data.images.keys())
    video_key = video_keys[0] if video_keys else None
    print(f"Found joints: {joint_names}")
    print(f"Found video key: {video_key}")
    
    del temp_policy
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 2: Collect ground truth actions
    print("\nCollecting ground truth actions...")
    gt_outputs = collect_ground_truth(
        episode_data, modality_config, embodiment_tag, joint_names, max_steps
    )
    print("  Done collecting ground truth.")
    
    # Step 3: Pre-generate noise for deterministic comparison
    print("\nPre-generating initial noise for deterministic comparison...")
    # Get model dimensions from a temporary policy load
    temp_policy, _ = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
    action_horizon = temp_policy.model.action_head.config.action_horizon
    action_dim = temp_policy.model.action_head.action_dim
    device = temp_policy.model.device
    dtype = torch.float32
    del temp_policy
    gc.collect()
    torch.cuda.empty_cache()
    
    # Generate noise list with fixed seed for reproducibility
    set_all_seeds(42)
    initial_noise_list = [
        torch.randn(1, action_horizon, action_dim, device=device, dtype=dtype)
        for _ in range(max_steps)
    ]
    print(f"  Generated {max_steps} noise tensors of shape (1, {action_horizon}, {action_dim})")
    
    # Step 4: Load original policy, collect outputs, then free memory
    print("\nLoading original policy...")
    set_all_seeds(42)
    original_policy, dataset = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
    time_start = time.time()
    original_outputs = collect_policy_outputs(
        original_policy, dataset, joint_names, max_steps, "Original",
        initial_noise_list=initial_noise_list
    )
    time_end = time.time()
    original_time_taken = time_end - time_start
    del original_policy
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed original policy memory.")
    
    comparison_outputs = None
    if not use_exported:
        # Step 5: Load modified policy, collect outputs, then free memory
        print("\nLoading modified policy...")
        set_all_seeds(42)
        modified_policy, dataset = get_policy_and_dataset(model_path, dataset_path, embodiment_tag, video_backend)
        set_all_seeds(42)
        modified_policy = make_modifications(modified_policy)
        time_start = time.time()
        comparison_outputs = collect_policy_outputs(
            modified_policy, dataset, joint_names, max_steps, "Modified",
            initial_noise_list=initial_noise_list
        )
        time_end = time.time()
        modified_time_taken = time_end - time_start
        del modified_policy, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("  Freed modified policy memory.")


    else:
        print("\nLoading exported policy...")

        exported_policy = InferenceManager(exported_model_path)
        # Get mock inputs to see the expected format
        mock_inputs = exported_policy.get_mock_input()
        print("run warm up the policy")
        for i in range(5):
            with torch.inference_mode():
                _ = exported_policy.run_policy(mock_inputs)
        print("  Warm up complete")

        time_start = time.time()
        comparison_outputs = collect_exported_policy_outputs(
            exported_policy, dataset, modality_config, embodiment_tag,
            joint_names, video_key, initial_noise_list, max_steps
        )
        time_end = time.time()
        modified_time_taken = time_end - time_start
        del exported_policy, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("  Freed exported policy memory.")
    
    # Step 7: Analyze and plot results
    print("\n" + "="*90)
    print("Analysis Results (per dimension, across all steps)")
    print("="*90)
    
    for joint_name in joint_names:
        gt_data = gt_outputs[joint_name]
        original_data = original_outputs[joint_name]
        exported_data = comparison_outputs[joint_name]
        
        # Compute and print statistics
        stats = compute_error_statistics(original_data, exported_data)
        print_error_statistics(joint_name, stats, max_steps)
        
        # Create plot (with optional exported data)
        plot_joint_comparison(joint_name, gt_data, original_data, exported_data, output_dir)
    
    if show_plots:
        plt.show()
    
    print(f"  Original policy:")
    print(f"  time taken: {original_time_taken} seconds")
    print(f"  time per step: {original_time_taken / max_steps} seconds")
    print(f"  Modified policy:")
    print(f"  time taken: {modified_time_taken} seconds")
    print(f"  time per step: {modified_time_taken / max_steps} seconds")


if __name__ == "__main__":
    # Set use_exported=True to include exported policy in comparison
    plot_policy_comparison(
        max_steps=args.max_steps,
        use_exported=args.use_exported,
        show_plots=args.show_plots,
        exported_model_path=args.model_yaml_path,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend
    )
