from .utils import video_transform_modules as export_video
from .utils import state_action_transform_modules as export_state_action
from .utils import concat_transform_modules as export_concat
from .utils import gr00t_transform_modules as export_gr00t_state_action
from .utils import gr00t_tokenizer as export_gr00t_video_language
from .utils import video_language_transfrom_module as export_eagle2_video_language

import torch
from .utils.export_utils import batch_tensorize_and_split
from .utils.export_utils import ComposedGr00tModule
from .utils.export_utils import test_gr00t_process_consistency
from .utils.export_utils import get_input_info
import os
import yaml


def export_and_test_preprocess(data, policy, model_path):
    # use different data to trace and test the model
    video_inputs, state_inputs, _, language_inputs = batch_tensorize_and_split(
        data)
    output_gr00t = dict(data)

    output_gr00t = get_input_info(policy, data)
    state_action_module = ComposedGr00tModule()
    video_module = ComposedGr00tModule()
    for idx, preprocessing_step in enumerate(policy._modality_transform.transforms):
        params = preprocessing_step.__dict__
        # Get the class name of the preprocessing step
        step_class_name = preprocessing_step.__class__.__name__
        state_action_step = None
        video_step = None
        eagle2_video_step = None
        # Import the corresponding export module class
        if "Video" in step_class_name:
            video_step = getattr(export_video, step_class_name)
        elif "StateAction" in step_class_name:
            state_action_step = getattr(export_state_action, step_class_name)
        elif "Concat" in step_class_name:
            state_action_step = getattr(export_concat, step_class_name)
            video_step = getattr(export_concat, step_class_name)
        elif "GR00TTransform" in step_class_name:
            state_action_step = getattr(
                export_gr00t_state_action, step_class_name)
            eagle2_video_step = getattr(
                export_eagle2_video_language, "Eagle2VideoTransform")
            metadata = {
                'default_instruction': params['default_instruction'],
                'embodiment_tag': params['embodiment_tag'].value,
                'embodiment_tag_mapping': params['embodiment_tag_mapping'],
                'apply_to': params['apply_to'],
                'formalize_language': params['formalize_language'],
            }
            with open(os.path.join(model_path, "tokenizer_params.yaml"), "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)
        else:
            print(f"Unknown module type: {step_class_name}")
            continue

        # Extract the initialization parameters from the preprocessing step

        if state_action_step:
            print(
                f"Creating {step_class_name} export instance and adding to state_action module")
            state_action_module.add_module(
                f"step_{idx}_{step_class_name}", state_action_step(**params))

        if video_step:
            print(
                f"Creating {step_class_name}export instance and adding to video module")
            video_module.add_module(
                f"step_{idx}_{step_class_name}", video_step(**params))

        if eagle2_video_step:
            from .utils.video_language_transfrom_module import Gr00tVideoLanguageTransform
            eagle2_tokenizer = Gr00tVideoLanguageTransform(**params)
            eagle2_tokenizer.set_gr00t_transform_outputs(output_gr00t)

    # new requirement: convert the input to f32 required
    state_inputs = {k: v.to(torch.float32)
                    for k, v in state_inputs.items()}

    traced_state_action_module = torch.jit.trace(
        state_action_module, state_inputs, strict=False)
    traced_state_action_module.save(os.path.join(
        model_path, "preprocess_state_action.pt"))
    traced_state_action_module = torch.jit.load(
        os.path.join(model_path, "preprocess_state_action.pt"))

    state_action_output_export = traced_state_action_module(
        state_inputs)

    traced_video_module = torch.jit.trace(
        video_module, video_inputs, strict=False)
    traced_video_module.save(os.path.join(
        model_path, "preprocess_video.pt"))
    traced_video_module = torch.jit.load(
        os.path.join(model_path, "preprocess_video.pt"))

    video_output = traced_video_module(video_inputs)

    video_language_inputs = {**video_output, **language_inputs}

    with open(os.path.join(model_path, "tokenizer_params.yaml"), "r") as f:
        tokenizer_params = yaml.load(f, Loader=yaml.FullLoader)
    video_language_module = export_gr00t_video_language.GR00TTransform(
        **tokenizer_params)

    python_gr00t_output = video_language_module(video_language_inputs)

    scripted_module = torch.jit.script(eagle2_tokenizer)
    scripted_module.save(os.path.join(
        model_path, "eagle2_tokenizer.pt"))
    scripted_module = torch.jit.load(os.path.join(
        model_path, "eagle2_tokenizer.pt"))
    video_language_output = scripted_module(video_output)

    # Combine video_output and state_action_output_export into a single dictionary
    output_export = {**state_action_output_export, **video_language_output}
    output_python_tokenizer = {
        **state_action_output_export, **python_gr00t_output}

    print("testing both implementations of the tokenizer")
    return (test_gr00t_process_consistency(output_export, output_gr00t) and
            test_gr00t_process_consistency(output_python_tokenizer, output_gr00t))


if __name__ == "__main__":
    from .utils.gr00t_policy_and_dataset import policy, dataset

    model_path = "saved_models"
    export_and_test_preprocess(dataset[0], policy, model_path)
