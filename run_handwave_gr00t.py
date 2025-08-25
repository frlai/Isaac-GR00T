from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
import hand_wave.utils as utils
from deployment_scripts.export_gr00t import export_gr00t


def get_policy_and_dataset(dataset_path: str,
                           model_path: str,
                           device: str = "cuda"):
    # load the policy
    data_config = utils.DATA_CONFIG_MAP["unitree_g1_v2"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    EMBODIMENT_TAG = "new_embodiment"
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
        video_backend="torchvision_av",  # Use PyAV for better AV1 codec support
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    return policy, dataset


if __name__ == "__main__":
    model_path = "nvidia/GR00T-N1.5-3B-WaveHand-Dev"
    dataset_path = "hand_wave/g1_wave"
    policy, dataset = get_policy_and_dataset(
        dataset_path=dataset_path,
        model_path=model_path,
    )

    data = dataset[5]
    for k, v in data.items():
        try:
            print(f"{k}, shape: {v.shape}, max: {v.max()}, min: {v.min()}")
        except Exception as e:
            print(f"{k}, value: {v}")

    print(policy.get_action(data))

    export_gr00t(policy, dataset, 'saved_models')
