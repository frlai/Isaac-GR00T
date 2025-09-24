import argparse
import os
from deployment_scripts.export_gr00t import get_policy_and_dataset, export_gr00t
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

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"Save model path: {args.save_model_path}")

    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path)
    export_gr00t(policy, dataset, args.save_model_path)

    # load policy again because export process may change some policy backends
    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path)

    print("****VALIDATING GR00T MODEL****")
    plot_action_distribution(
        policy, dataset, plot_actions=["action.left_arm",
                                       "action.right_arm",
                                       "action.left_hand",
                                       "action.right_hand"],
        output_dir=os.path.join(os.getcwd(), "gr00t_models", "plots", "python"))

    print("****PROFILING EXPORTED MODEL")
    print("****RESULTS MAY TAKE TIME IF YOU DON'T HAVE THE GPU EXECUTION PROVIDER IN ONNXRUNTIME****")
    runner = ExportedGr00tRunner(os.path.join(os.getcwd(), "gr00t_models"))
    plot_action_distribution(
        runner, dataset, plot_actions=["action.left_arm",
                                       "action.right_arm",
                                       "action.left_hand",
                                       "action.right_hand"],
        output_dir=os.path.join(os.getcwd(), "gr00t_models", "plots", "onnx"))

    annotate.start(name="gr00t_models", save_path=os.getcwd())
    resutls = runner.get_action(dataset[0])
    annotate.stop()
    annotate.compile_graph()
