import torch
import os
from .utils import concat_transform_modules as export_concat
from .utils import state_action_transform_modules as export_state_action
from .utils import gr00t_transform_modules as export_gr00t_state_action

from .utils.export_utils import ComposedGr00tModule
from .utils.export_utils import test_gr00t_process_consistency
from .utils.export_utils import describe_io
from gr00t.data.transform.base import InvertibleModalityTransform
import yaml


def get_unprocessed_action(policy, observations):
    from gr00t.model.policy import unsqueeze_dict_values

    # let the get_action handles both batch and single input
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    # Apply transforms
    normalized_input = policy.apply_transforms(observations)
    normalized_action = policy._get_action_from_normalized_input(
        normalized_input)

    return normalized_action


def export_and_test_postprocess(data, policy, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_gr00t = get_unprocessed_action(policy, data)
    output_export = {"action": output_gr00t.clone().detach().to(device)}

    postprocess_modules = ComposedGr00tModule()
    for idx, postprocessing_step in enumerate(reversed(policy.modality_transform.transforms)):
        params = postprocessing_step.__dict__
        # Get the class name of the preprocessing step
        step_class_name = postprocessing_step.__class__.__name__
        params["backward"] = True
        if isinstance(postprocessing_step, InvertibleModalityTransform):
            if "GR00TTransform" in step_class_name:
                step = getattr(export_gr00t_state_action, step_class_name)
            elif "Concat" in step_class_name:
                step = getattr(export_concat, step_class_name)
            if "StateAction" in step_class_name:
                step = getattr(export_state_action, step_class_name)

            print("exporting", step_class_name)
            postprocess_modules.add_module(
                f"step_{idx}_{step_class_name}", step(**params))

    # new requirement: convert the input to f32 required
    output_export = {k: v.to(torch.float16)
                     for k, v in output_export.items()}

    traced_postprocess_modules = torch.jit.trace(
        postprocess_modules, output_export, strict=False)
    traced_postprocess_modules.save(
        os.path.join(model_path, "postprocess_modules.pt"))
    traced_postprocess_modules = torch.jit.load(
        os.path.join(model_path, "postprocess_modules.pt"))

    input_description, input_format = describe_io(output_export)
    output_export = traced_postprocess_modules(output_export)
    output_description, output_format = describe_io(output_export)
    describe_postprocess_inputs = {"inference": {"input_nodes": input_description,
                                                 "output_nodes": output_description,
                                                 "input_format": [input_format],
                                                 "output_format": output_format}}
    yaml.dump(describe_postprocess_inputs, open(
        model_path+"/postprocess_modules.yaml", "w"))

    output_gr00t = policy._get_unnormalized_action(output_gr00t)

    return test_gr00t_process_consistency(output_export, output_gr00t)


if __name__ == "__main__":
    from .utils.gr00t_policy_and_dataset import policy, dataset

    export_and_test_postprocess(dataset[0], policy, "saved_models")
