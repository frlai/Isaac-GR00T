import os
import torch
import onnxruntime
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from deployment_scripts.export_scripts.utils.export_utils import batch_tensorize_and_split
import sys
from leapp import annotate

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
    from deployment_scripts.export_scripts.utils.export_utils import get_input_info
    from deployment_scripts.export_scripts.export_denoising_subgraph_onnx import export_denoising_subgraph
    from deployment_scripts.export_scripts.export_postprocess import export_and_test_postprocess
    from deployment_scripts.export_scripts.export_preprocess import export_and_test_preprocess
    from deployment_scripts.export_scripts.export_onnx import export_onnx
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


class ExportedGr00tRunner:
    def __init__(self, save_model_path: str):
        self.save_model_path = save_model_path
        self.preprocess_video_path = os.path.join(
            save_model_path, "preprocess", "preprocess_video.pt")
        self.preprocess_state_action_path = os.path.join(
            save_model_path, "preprocess", "preprocess_state_action.pt")
        self.eagle_2_tokenizer_path = os.path.join(
            save_model_path, "preprocess", "eagle2_tokenizer.pt")
        self.postprocess_modules_path = os.path.join(
            save_model_path, "postprocess", "postprocess_modules.pt")
        self.vit_path = os.path.join(
            save_model_path, "eagle2", "vit.onnx")
        self.llm_path = os.path.join(
            save_model_path, "eagle2", "llm.onnx")
        self.denoising_subgraph_path = os.path.join(
            save_model_path, "action_head", "denoising_subgraph.onnx")

        self._preprocess_video = torch.jit.load(self.preprocess_video_path)
        self._preprocess_state_action = torch.jit.load(
            self.preprocess_state_action_path)
        self._eagle_2_tokenizer = torch.jit.load(self.eagle_2_tokenizer_path)
        self._postprocess_modules = torch.jit.load(
            self.postprocess_modules_path)
        # Set up execution providers for GPU acceleration
        # Force GPU execution only - fail fast if GPU unavailable
        providers = ['CUDAExecutionProvider']

        self._vit = onnxruntime.InferenceSession(
            self.vit_path, providers=providers)
        self._llm = onnxruntime.InferenceSession(
            self.llm_path, providers=providers)
        self._denoising_subgraph = onnxruntime.InferenceSession(
            self.denoising_subgraph_path, providers=providers)

        # Apply decorators with backend_params after initialization
        self._apply_decorators()

    def _apply_decorators(self):
        """Apply decorators with backend_params containing model paths"""
        # Apply decorator to preprocess_state_action
        self.preprocess_state_action = annotate.method(
            backend_params={"model_path": self.preprocess_state_action_path}
        )(self.preprocess_state_action)

        # Apply decorator to preprocess_video
        self.preprocess_video = annotate.method(
            backend_params={"model_path": self.preprocess_video_path}
        )(self.preprocess_video)

        # Apply decorator to eagle_2_tokenizer
        self.eagle_2_tokenizer = annotate.method(
            node_name="tokenizer",
            backend_params={"model_path": self.eagle_2_tokenizer_path}
        )(self.eagle_2_tokenizer)

        # Apply decorator to vit
        self.vit = annotate.method(
            node_name="vit_engine",
            backend_params={"model_path": self.vit_path}
        )(self.vit)

        # Apply decorator to llm
        self.llm = annotate.method(
            node_name="llm_engine",
            backend_params={"model_path": self.llm_path}
        )(self.llm)

        # Apply decorator to denoising_subgraph
        self.denoising_subgraph = annotate.method(
            backend_params={"model_path": self.denoising_subgraph_path}
        )(self.denoising_subgraph)

        # Apply decorator to postprocess_modules
        self.postprocess_modules = annotate.method(
            backend_params={"model_path": self.postprocess_modules_path}
        )(self.postprocess_modules)

    def preprocess_state_action(self, state_inputs):
        preprocessed_state = self._preprocess_state_action(state_inputs)
        return preprocessed_state

    def preprocess_video(self, video_inputs):
        preprocessed_video = self._preprocess_video(video_inputs)
        return preprocessed_video

    def eagle_2_tokenizer(self, eagle_2_tokenizer_inputs):
        eagle_2_tokenizer_outputs = self._eagle_2_tokenizer(
            eagle_2_tokenizer_inputs)
        return eagle_2_tokenizer_outputs

    def vit(self, eagle_pixel_values):
        vit_inputs = {
            "pixel_values": eagle_pixel_values.cpu().numpy(),
        }
        vit_embeds = self._vit.run(None, vit_inputs)[0]
        vit_embeds = torch.from_numpy(vit_embeds)
        return vit_embeds

    def llm(self, eagle_input_ids, eagle_attention_mask, vit_embeds):
        llm_inputs = {
            "input_ids": eagle_input_ids.cpu().numpy(),
            "attention_mask": eagle_attention_mask.cpu().numpy(),
            "vit_embeds": vit_embeds.cpu().numpy(),
        }
        embeddings = self._llm.run(None, llm_inputs)[0]
        embeddings = torch.from_numpy(embeddings)
        return embeddings

    def denoising_subgraph(self, embeddings, state, embodiment_id):
        denoising_subgraph_inputs = {
            "embeddings": embeddings.cpu().numpy(),
            "state": state.cpu().numpy(),
            "embodiment_id": embodiment_id.cpu().numpy(),
        }
        action = self._denoising_subgraph.run(
            None, denoising_subgraph_inputs)[0]
        action = torch.from_numpy(action)
        action = {
            "action": action.cuda(),
        }
        return action

    def postprocess_modules(self, action):
        action = self._postprocess_modules(action)
        return action

    def get_action(self, data):
        video_inputs, state_inputs, _, _ = batch_tensorize_and_split(data)
        for k in video_inputs:
            video_inputs[k] = video_inputs[k].to(torch.float32)
        for k in state_inputs:
            state_inputs[k] = state_inputs[k].to(torch.float32)

        preprocessed_state = self.preprocess_state_action(state_inputs)
        preprocessed_video = self.preprocess_video(video_inputs)
        eagle_2_tokenizer_outputs = self.eagle_2_tokenizer(
            preprocessed_video)
        vit_embeds = self.vit(eagle_2_tokenizer_outputs['eagle_pixel_values'])
        embeddings = self.llm(eagle_2_tokenizer_outputs['eagle_input_ids'],
                              eagle_2_tokenizer_outputs['eagle_attention_mask'],
                              vit_embeds)
        action = self.denoising_subgraph(
            embeddings, preprocessed_state['state'], eagle_2_tokenizer_outputs['embodiment_id'])
        postprocess_modules_outputs = self.postprocess_modules(action)

        return postprocess_modules_outputs
