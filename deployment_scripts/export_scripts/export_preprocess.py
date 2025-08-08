from .utils import video_transform_modules as export_video
from .utils import state_action_transform_modules as export_state_action
from .utils import concat_transform_modules as export_concat
from .utils import gr00t_transform_modules as export_gr00t_state_action
from .utils import gr00t_tokenizer as export_gr00t_video_language

import torch
import copy
from .utils.export_utils import batch_tensorize_and_split
from .utils.export_utils import ComposedGr00tModule
from .utils.export_utils import describe_io, test_gr00t_process_consistency
import os
import yaml


def get_preprocessed_data(policy, observations):
    from gr00t.model.policy import unsqueeze_dict_values

    # let the get_action handles both batch and single input
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)
    retval = policy.apply_transforms(observations)
    return retval


def export_and_test_preprocess(data, policy, models_base_path):
    # use different data to trace and test the model
    video_inputs, state_inputs, action_inputs, language_inputs = batch_tensorize_and_split(
        data)
    output_gr00t = dict(data)

    output_gr00t = get_preprocessed_data(policy, data)
    state_action_module = ComposedGr00tModule()
    video_module = ComposedGr00tModule()
    for idx, preprocessing_step in enumerate(policy._modality_transform.transforms):
        params = preprocessing_step.__dict__
        # Get the class name of the preprocessing step
        step_class_name = preprocessing_step.__class__.__name__
        state_action_step = None
        video_step = None
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
            # import pdb
            # pdb.set_trace()
            # vlm_processor = params["vlm_processor"]
            # metadata = {
            #     "image_size": vlm_processor.image_size,
            #     "context_len": vlm_processor.context_len,
            #     "per_tile_len": vlm_processor.per_tile_len,
            #     "model_spec": vlm_processor.model_spec.__dict__,
            #     "max_input_tiles": vlm_processor.max_input_tiles,
            #     "norm_type": vlm_processor.norm_type,
            # }
            # params["vlm_processor_metadata"] = metadata
            # with open(models_base_path+"/tokenizer_params.yaml", "w") as f:
            #     save_params = copy.deepcopy(params)
            #     del save_params["embodiment_tag"]
            #     del save_params["vlm_processor"]
            #     print(save_params)
            #     yaml.dump(save_params, f, default_flow_style=False)

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

    # new requirement: convert the input to f32 required
    state_inputs = {k: v.to(torch.float32)
                    for k, v in state_inputs.items()}

    traced_state_action_module = torch.jit.trace(
        state_action_module, state_inputs, strict=False)
    traced_state_action_module.save(os.path.join(
        models_base_path, "preprocess_state_action.pt"))
    traced_state_action_module = torch.jit.load(
        os.path.join(models_base_path, "preprocess_state_action.pt"))

    input_description, input_format = describe_io(state_inputs)
    state_action_output_export = traced_state_action_module(
        state_inputs)
    output_description, output_format = describe_io(state_action_output_export)
    describe_state_action_inputs = {"inference": {"input_nodes": input_description,
                                                  "output_nodes": output_description,
                                                  "input_format": [input_format],
                                                  "output_format": output_format}}
    yaml.dump(describe_state_action_inputs, open(
        models_base_path+"/preprocess_state_action.yaml", "w"))

    traced_video_module = torch.jit.trace(
        video_module, video_inputs, strict=False)
    traced_video_module.save(os.path.join(
        models_base_path, "preprocess_video.pt"))
    traced_video_module = torch.jit.load(
        os.path.join(models_base_path, "preprocess_video.pt"))

    input_description, input_format = describe_io(video_inputs)
    video_output = traced_video_module(video_inputs)
    output_description, output_format = describe_io(video_output)
    describe_video_inputs = {"inference": {"input_nodes": input_description,
                                           "output_nodes": output_description,
                                           "input_format": [input_format],
                                           "output_format": output_format}}
    yaml.dump(describe_video_inputs, open(
        models_base_path+"/preprocess_video.yaml", "w"))

    # with open(models_base_path+"/tokenizer_params.yaml", "r") as f:
    #     params = yaml.load(f, Loader=yaml.FullLoader)
    # video_language_module = export_gr00t_video_language.GR00TTransform(
    #     **params)

    # video_language_inputs = {**video_output, **language_inputs}
    # input_description, input_format = describe_io(video_language_inputs)

    # video_language_output = video_language_module(video_language_inputs)
    # output_description, output_format = describe_io(video_language_output)
    # describe_video_language_inputs = {"inference": {"input_nodes": input_description,
    #                                                 "output_nodes": output_description,
    #                                                 "input_format": [input_format],
    #                                                 "output_format": output_format}}
    # yaml.dump(describe_video_language_inputs, open(
    #     models_base_path+"/preprocess_video_language.yaml", "w"))

    # Combine video_output and state_action_output_export into a single dictionary
    # output_export = {**state_action_output_export, **video_language_output}
    output_export = state_action_output_export

    return test_gr00t_process_consistency(output_export, output_gr00t)


if __name__ == "__main__":
    from .utils.gr00t_policy_and_dataset import policy, dataset

    model_path = "saved_models"
    export_and_test_preprocess(dataset[0], policy, model_path)
