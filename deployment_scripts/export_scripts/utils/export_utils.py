import time
import torch
import numpy as np


# Helper functions
def unsqueeze_dict_values(data: dict[str, any]) -> dict[str, any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data):
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data


def check_state_is_batched(obs):
    for k, v in obs.items():
        if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
            return False
    return True


def map_from_torch_dtype(dtype):
    if dtype == torch.float32:
        return "kFloat32"
    elif dtype == torch.float16:
        return "kFloat16"
    elif dtype == torch.int32:
        return "kInt32"
    elif dtype == torch.int64:
        return "kInt64"
    elif dtype == torch.uint8:
        return "kUInt8"
    elif dtype == torch.int8:
        return "kInt8"
    elif dtype == torch.bool:
        return "kBool"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def describe_io_helper(data, name_str):
    data_description = {}
    if isinstance(data, list):
        format = []
        for idx, item in enumerate(data):
            child_description, child_format = describe_io_helper(
                item, name_str+str(idx))
            format.append(child_format)
            data_description = {**data_description, **child_description}
        return data_description, format
    elif isinstance(data, dict):
        format = {}
        for k, v in data.items():
            child_description, child_format = describe_io_helper(v, name_str+k)
            format[k] = child_format
            data_description = {**data_description, **child_description}
    elif isinstance(data, torch.Tensor):
        data_description[name_str] = {
            "dtype": map_from_torch_dtype(data.dtype),
            "dim": " ".join(str(int(x)) for x in data.shape)
        }
        format = name_str
    else:
        data_description[name_str] = type(data)
        format = name_str

    return data_description, format


def describe_io(data):
    data_description = {}
    format = None
    data_description, format = describe_io_helper(data, "")
    return data_description, format

# Convert all non-tensor values to tensors


def convert_all_to_tensor(data, device='cuda'):
    tensorized_data = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            try:
                if hasattr(value, '__array__') or hasattr(value, 'numpy'):  # For numpy arrays
                    tensorized_data[key] = torch.tensor(value, device=device)
                elif isinstance(value, (int, float, bool)):
                    tensorized_data[key] = torch.tensor(value, device=device)
                elif isinstance(value, list):
                    tensorized_data[key] = torch.tensor(value, device=device)
                else:
                    # Try generic conversion
                    tensorized_data[key] = torch.tensor(value, device=device)
            except Exception:
                print(
                    f"Warning: Could not convert '{key}' to tensor.")
                tensorized_data[key] = value
        else:
            # Already a tensor
            tensorized_data[key] = value

    return tensorized_data


def batch_tensorize_and_split(data):
    # batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batched = check_state_is_batched(data)
    if not batched:
        data = unsqueeze_dict_values(data)

    # tensorize
    tensorized_data = convert_all_to_tensor(data, device)

    # split
    video_inputs = {k: v for k, v in tensorized_data.items(
    ) if k.startswith('video.')}
    state_inputs = {k: v for k, v in tensorized_data.items(
    ) if k.startswith('state.')}
    action_inputs = {k: v for k, v in tensorized_data.items(
    ) if k.startswith('action.')}
    language_inputs = {k: v for k, v in tensorized_data.items(
    ) if k.startswith('annotation.')}

    return video_inputs, state_inputs, action_inputs, language_inputs


def verify_preprocess(output_export, output_gr00t, name):
    # verify video
    success = True
    # Check if the tensors are on the same device
    if output_export.device != output_gr00t.device:
        # Move one tensor to match the device of the other for comparison
        output_gr00t = output_gr00t.to(output_export.device)

    if output_export.shape != output_gr00t.shape:
        print(
            f"The tensors {name} are not the same shape! "
            f"{output_export.shape} != {output_gr00t.shape}")
        return False
    if output_export.dtype != output_gr00t.dtype:
        print(
            f"Warning: The tensors {name} are not the same dtype! "
            f"{output_export.dtype} != {output_gr00t.dtype}")
    # Check if the tensors are equal
    if torch.all(output_export == output_gr00t):
        print(f"The tensors {name} are exactly equal!")
        success &= True
    else:
        # Calculate the mean absolute difference
        mean_diff = torch.abs(
            output_export - output_gr00t).mean()
        print(
            f"The tensors {name} are not exactly equal."
            f" Mean absolute difference: {mean_diff}")
        # Calculate the maximum absolute difference
        max_diff = torch.abs(output_export - output_gr00t).max()
        min_diff = torch.abs(output_export - output_gr00t).min()
        print(
            f"Maximum absolute difference for {name}: {max_diff}, "
            f"Min diff is {min_diff}")
        # Check if they're close (within a small tolerance)
        if torch.allclose(output_export, output_gr00t.to(output_export.dtype), rtol=1e-2, atol=1e-2):
            print(
                f"The tensors {name} are approximately equal (within tolerance).")
        else:
            success &= False

    return success


def test_export(original_model, exported_model, *args,
                network_accuracy_test=True,
                network_consistency_test=5,
                network_speed_test=0.05):

    if network_consistency_test:
        test_network_consistency(
            exported_model, network_consistency_test, *args)
        print(
            f"network consistency test passed over {network_consistency_test} runs")
    if network_accuracy_test:
        test_network_output(exported_model, original_model, *args)
    if network_speed_test:

        (time_exported,
         time_original) = test_network_speed(exported_model, original_model, *args,
                                             time_difference_threshold=network_speed_test)
        print(
            f"network speed test passed, time_exported={time_exported}, "
            f"time_original={time_original}")


def test_network_consistency(model, runs, *args):
    prev_backbone_output = None
    failures = 0
    for i in range(runs):
        failure = False
        d_backbone_output = model(*args)
        if prev_backbone_output is not None:
            for key in prev_backbone_output.keys():
                if not torch.allclose(prev_backbone_output[key], d_backbone_output[key]):
                    print(
                        f"Test {i+1}/30: {key} failed")
                    print(
                        f"original output: shape={prev_backbone_output[key].shape}, "
                        f"{prev_backbone_output[key]}")
                    print(
                        f"exported output: shape={d_backbone_output[key].shape}, "
                        f"{d_backbone_output[key]}")
                    failure = True

        failures += failure
        prev_backbone_output = d_backbone_output

    if failures > 0:
        raise AssertionError(f"Failed {failures}/{runs} tests")
    else:
        return prev_backbone_output


def test_network_output(exported_model, original_model, *args):
    output_original = original_model(*args)
    output_exported = exported_model(*args)

    for key in output_original.keys():
        if not torch.equal(output_original[key], output_exported[key]):
            raise AssertionError(f"Failed because key difference: {key}, "
                                 f"\n\noriginal output \n\n{output_original[key]}"
                                 f"\n\nexported output \n\n{output_exported[key]}")


def test_network_speed(exported_model, original_model, *args,
                       time_difference_threshold=0.1, runs=5):
    start_time = time.time()
    for i in range(runs):
        _ = original_model(*args)
    end_time = time.time()
    time_original = (end_time - start_time) / runs

    start_time = time.time()
    for i in range(runs):
        _ = exported_model(*args)
    end_time = time.time()
    time_exported = (end_time - start_time) / runs

    if time_exported > time_original * (1 + time_difference_threshold):
        raise AssertionError(
            f"Exported model is slower than original model by "
            f"{time_exported - time_original} seconds")
    else:
        return time_exported, time_original


def test_gr00t_process_consistency(output_export, output_gr00t):
    success = True
    for key in output_export.keys():
        if key not in output_gr00t.keys():
            print(f"\nKey {key} not found in output_gr00t\n\n")
            continue

        print(f'comparing: {key}')
        if not isinstance(output_gr00t[key], torch.Tensor):
            if isinstance(output_gr00t[key], np.ndarray):
                # Print the dtype of output_gr00t
                output_gr00t_tensor = torch.from_numpy(output_gr00t[key])
            elif isinstance(output_gr00t[key], int):
                output_gr00t_tensor = torch.tensor(
                    output_gr00t[key], dtype=torch.int64)
        else:
            output_gr00t_tensor = output_gr00t[key]
        success &= verify_preprocess(output_export[key],
                                     output_gr00t_tensor,
                                     key)
    if success:
        print("SUCCESS")
    else:
        print("FAILED")

    return success


def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    # Apply transforms
    normalized_input = policy.apply_transforms(observations)
    return normalized_input


class ComposedGr00tModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, **kwargs):
        result = data
        for name, submodule in self._modules.items():
            result = submodule(result)
        return result
