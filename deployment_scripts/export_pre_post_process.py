from export_scripts.export_preprocess import export_and_test_preprocess
from export_scripts.export_postprocess import export_and_test_postprocess
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import LeRobotSingleDataset
import torch
import os
if __name__ == "__main__":

    model_path = "nvidia/GR00T-N1.5-3B"
    dataset_path = "demo_data/robot_sim.PickNPlace"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "saved_models"
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

    step_data = dataset[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    export_and_test_preprocess(
        step_data, policy, os.path.join(save_path, "preprocess"))
    export_and_test_postprocess(
        step_data, policy, os.path.join(save_path, "postprocess"))
