import argparse
import os
import torch
import onnxruntime
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from export_scripts.utils.export_utils import batch_tensorize_and_split
import sys

# Ensure export_scripts can be imported as a package from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
deployment_scripts_dir = current_dir
if deployment_scripts_dir not in sys.path:
    sys.path.insert(0, deployment_scripts_dir)


def get_policy_and_dataset(dataset_path: str, model_path: str, device: str = "cuda"):
    # load the policy
    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    EMBODIMENT_TAG = "gr1"
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    modality_config = policy.modality_config
    # load the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    return policy, dataset


def export_gr00t(policy: Gr00tPolicy, dataset: LeRobotSingleDataset, onnx_model_path: str):
    from export_scripts.utils.export_utils import get_input_info
    from export_scripts.export_denoising_subgraph_onnx import export_denoising_subgraph
    from export_scripts.export_postprocess import export_and_test_postprocess
    from export_scripts.export_preprocess import export_and_test_preprocess
    from export_scripts.export_onnx import export_onnx
    # export the preprocess model
    preprocess_model_path = os.path.join(
        onnx_model_path, "preprocess")
    if not os.path.exists(preprocess_model_path):
        os.makedirs(preprocess_model_path)
    export_and_test_preprocess(
        data=dataset[0],
        policy=policy,
        model_path=preprocess_model_path,
    )

    # export the postprocess model
    postprocess_model_path = os.path.join(
        onnx_model_path, "postprocess")
    if not os.path.exists(postprocess_model_path):
        os.makedirs(postprocess_model_path)
    export_and_test_postprocess(
        data=dataset[0],
        policy=policy,
        model_path=postprocess_model_path,
    )
    inputs = get_input_info(policy, dataset[0])

    attention_mask = inputs["eagle_attention_mask"]
    state = inputs["state"]

    # export the onnx model
    export_onnx(
        dataset=dataset,
        policy=policy,
        input_state=state,
        attention_mask=attention_mask,
        onnx_model_path=onnx_model_path,
    )

    # export the denoising subgraph
    export_denoising_subgraph(
        policy=policy,
        input_state=state,
        attention_mask=attention_mask,
        save_model_path=os.path.join(onnx_model_path, "action_head"),
    )


def run_exported_gr00t(dataset: LeRobotSingleDataset, save_model_path: str,
                       video_inputs, state_inputs):
    preprocess_video_path = os.path.join(
        save_model_path, "preprocess", "preprocess_video.pt")
    preprocess_state_action_path = os.path.join(
        save_model_path, "preprocess", "preprocess_state_action.pt")
    eagle_2_tokenizer_path = os.path.join(
        save_model_path, "preprocess", "eagle2_tokenizer.pt")
    postprocess_modules_path = os.path.join(
        save_model_path, "postprocess", "postprocess_modules.pt")

    vit_path = os.path.join(save_model_path, "eagle2", "vit.onnx")
    llm_path = os.path.join(save_model_path, "eagle2", "llm.onnx")

    denoising_subgraph_path = os.path.join(
        save_model_path, "action_head", "denoising_subgraph.onnx")

    preprocess_video = torch.jit.load(preprocess_video_path)
    preprocess_state_action = torch.jit.load(preprocess_state_action_path)
    eagle_2_tokenizer = torch.jit.load(eagle_2_tokenizer_path)
    postprocess_modules = torch.jit.load(postprocess_modules_path)

    vit = onnxruntime.InferenceSession(vit_path)
    llm = onnxruntime.InferenceSession(llm_path)
    denoising_subgraph = onnxruntime.InferenceSession(denoising_subgraph_path)

    preprocessed_state = preprocess_state_action(state_inputs)

    preprocessed_video = preprocess_video(video_inputs)

    eagle_2_tokenizer_outputs = eagle_2_tokenizer(preprocessed_video)

    vit_inputs = {
        "pixel_values": eagle_2_tokenizer_outputs["eagle_pixel_values"].cpu().numpy()
    }

    vit_outputs = vit.run(None, vit_inputs)
    llm_inputs = {
        "vit_embeds": vit_outputs[0],
        "attention_mask": eagle_2_tokenizer_outputs["eagle_attention_mask"].cpu().numpy(),
        "input_ids": eagle_2_tokenizer_outputs["eagle_input_ids"].cpu().numpy(),
    }
    llm_outputs = llm.run(None, llm_inputs)

    denoising_subgraph_inputs = {
        "embeddings": llm_outputs[0],
        "state": preprocessed_state["state"].cpu().numpy(),
        "embodiment_id": eagle_2_tokenizer_outputs["embodiment_id"].cpu().numpy(),
    }

    denoising_subgraph_outputs = denoising_subgraph.run(
        None, denoising_subgraph_inputs)

    postprocess_modules_inputs = {
        "action": torch.from_numpy(
            denoising_subgraph_outputs[0]).cuda()
    }
    postprocess_modules_outputs = postprocess_modules(
        postprocess_modules_inputs)

    return postprocess_modules_outputs


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
        default=os.path.join(os.getcwd(), "saved_models"),
    )

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"Save model path: {args.save_model_path}")

    policy, dataset = get_policy_and_dataset(
        args.dataset_path, args.model_path)
    video_inputs, state_inputs, _, _ = batch_tensorize_and_split(
        dataset[0])
    export_gr00t(policy, dataset, args.save_model_path)
    resutls = run_exported_gr00t(
        dataset, args.save_model_path, video_inputs, state_inputs)
