import os
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
# from gr00t.experiment.data_config import DATA_CONFIG_MAP
from utils import DATA_CONFIG_MAP
from deployment_scripts.export_gr00t import export_gr00t
from deployment_scripts.export_gr00t import ExportedGr00tRunner
from deployment_scripts.export_scripts.verification import plot_action_distribution
from leapp import annotate


def get_policy_and_dataset(dataset_path: str = '/home/binliu/groot/dataset/g1_wave',
                           model_path: str = 'nvidia/GR00T-N1.5-3B-WaveHand-Dev',
                           device: str = "cuda"):
    # load the policy
    data_config = DATA_CONFIG_MAP["unitree_g1_v2"]
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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'g1_wave')

    policy, dataset = get_policy_and_dataset(
        dataset_path=dataset_path,
        model_path='nvidia/GR00T-N1.5-3B-WaveHand-Dev',
        device='cuda',
    )
    print("****EXPORTING GR00T MODEL****")
    export_gr00t(policy, dataset, "handwave_model")

    # load policy again because export process may change some policy backends
    policy, dataset = get_policy_and_dataset(
        dataset_path=dataset_path,
        model_path='nvidia/GR00T-N1.5-3B-WaveHand-Dev',
        device='cuda',
    )

    print("****VALIDATING GR00T MODEL****")
    plot_action_distribution(
        policy, dataset, plot_actions=["action.upper_body", "action.hands"], output_dir="./handwave_model/plots/python")

    print("****PROFILING EXPORTED MODEL")
    print("****RESULTS MAY TAKE TIME IF YOU DON'T HAVE THE GPU EXECUTION PROVIDER IN ONNXRUNTIME****")
    runner = ExportedGr00tRunner("handwave_model")
    plot_action_distribution(
        runner, dataset, plot_actions=["action.upper_body", "action.hands"], output_dir="./handwave_model/plots/onnx")

    annotate.start(name="handwave_model")
    runner.get_action(dataset[0])
    annotate.stop()
    annotate.compile_graph()
