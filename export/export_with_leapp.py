"""
GR00T model export using leapp framework.

This module handles exporting the GR00T policy to ONNX format using the leapp
tracing and export framework.
"""

from utils import get_policy_and_dataset, get_gr00t_input
from policy_modifications import make_modifications, get_action_traceable
import os
import gr00t
from leapp import annotate

import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default='nvidia/GR00T-N1.6-3B')
args.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/gr1.PickNPlace"))
args.add_argument("--embodiment_tag", type=str, default='gr1')
args.add_argument("--video_backend", type=str, default='torchcodec')
args.add_argument("--output_name", type=str, default='exported_gr00t')
args = args.parse_args()


def export_gr00t_with_leapp(policy, data, output_name='exported_gr00t'):
    """
    Export GR00T policy using leapp framework.
    
    Args:
        policy: Gr00tPolicy instance (will be modified in-place)
        data: Sample input data for tracing
        output_name: Name for the exported model
    """
    # Apply modifications to make policy traceable
    policy = make_modifications(policy)

    # Configure backbone export
    policy.model.backbone.forward = annotate.method(
        node_name='backbone',
        export_with='onnx-torchscript',
    )(policy.model.backbone.forward)

    # Configure action head export
    policy.model.action_head.get_action = annotate.method(
        node_name='action_head',
        export_with='onnx',
    )(policy.model.action_head.get_action)

    # Run tracing
    annotate.start(output_name, patch_numpy=False, dry_run=False)
    get_action_traceable(policy, data)
    annotate.stop()
    
    # Compile and export
    annotate.compile_graph(validate=False) # validate with comparison script
    
    print(f"Export completed: {output_name}")


def main():
    """Main entry point for export."""
    # Load policy and dataset
    policy, dataset = get_policy_and_dataset(model_path = args.model_path, 
                                            dataset_path = args.dataset_path, 
                                            embodiment_tag = args.embodiment_tag, video_backend = args.video_backend)
                        
    # Get sample input data
    data = get_gr00t_input(dataset, policy, step_index=0, step=None)
    # Export
    export_gr00t_with_leapp(policy, data, output_name=args.output_name)


if __name__ == "__main__":
    main()
