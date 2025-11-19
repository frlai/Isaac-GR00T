import argparse
import os
from deployment_scripts.export_gr00t import export_gr00t, get_policy_and_dataset
from deployment_scripts.export_gr00t import ExportedGr00tRunner
from deployment_scripts.export_scripts.verification import plot_action_distribution
from leapp import annotate

if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Path where the exported artifacts will be stored",
        default=os.path.join(os.getcwd(), "gr00t_models"),
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model",
        default="gr1",
    )

    parser.add_argument(
        "--modality_config",
        type=str,
        help="The modality config for the model",
        default="fourier_gr1_arms_only",
    )

    parser.add_argument(
        "--video_backend",
        choices=["decord", "torchvision_av"],
        type=str,
        help="The video backend for the model",
        default="decord",
    )

    parser.add_argument(
        "--plot_actions",
        type=str,
        nargs='+',
        help="The actions to plot (e.g., action.single_arm action.gripper)",
        default=[],
    )

    parser.add_argument(
        "--num_plot_steps",
        type=int,
        help="The number of steps to plot",
        default=30,
    )

    args = parser.parse_args()

    print("~~~~~~~~~~~~~~~~~~~~~ export_settings ~~~~~~~~~~~~~~~~~~~~~")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"Save model path: {args.save_model_path}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Modality config: {args.modality_config}")
    print(f"Video backend: {args.video_backend}")
    if len(args.plot_actions) != 0 and len(args.plot_actions) != 0:
        print(
            f"verifying export fidelity using actions: {args.plot_actions} using {args.num_plot_steps} steps")
    print("~~~~~~~~~~~~~~~~~~~~~ export_settings ~~~~~~~~~~~~~~~~~~~~~")

    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path, args.embodiment_tag, args.modality_config, video_backend=args.video_backend)
    export_gr00t(policy, dataset, args.save_model_path)

    # load policy again because export process may change some policy backends
    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path, args.embodiment_tag, args.modality_config, video_backend=args.video_backend)

    runner = ExportedGr00tRunner(
        os.path.join(os.getcwd(), args.save_model_path))
    if len(args.plot_actions) != 0:
        print("****VALIDATING GR00T MODEL****")
        plot_action_distribution(
            policy, dataset, plot_actions=args.plot_actions,
            output_dir=os.path.join(
                os.getcwd(), args.save_model_path, "plots", "python"),
            iters=args.num_plot_steps)

        print("****PROFILING EXPORTED MODEL")
        print("****RESULTS MAY TAKE TIME IF YOU DON'T HAVE THE GPU EXECUTION PROVIDER IN ONNXRUNTIME****")
        plot_action_distribution(
            runner, dataset, plot_actions=args.plot_actions,
            output_dir=os.path.join(
                os.getcwd(), args.save_model_path, "plots", "onnx"),
            iters=args.num_plot_steps)

    annotate.start(name=args.save_model_path)
    resutls = runner.get_action(dataset[0])
    annotate.stop()
    annotate.compile_graph()
