from typing import Any, cast


import kornia.augmentation as K
import torch

from stac_model.base import DataType


def normalize_dtype(torch_dtype: torch.dtype) -> DataType:
    """
    Convert a PyTorch dtype (e.g., torch.float32) to a standardized DataType.
    """
    return cast(DataType, str(torch_dtype).rsplit(".", 1)[-1])


def find_tensor_by_key(state_dict: dict[str, torch.Tensor], key_substring: str, reverse: bool = False) -> torch.Tensor:
    """
    Find a tensor in the state_dict by a substring in its key.
    If `reverse` is True, search from the end of the dictionary.
    """
    items = reversed(state_dict.items()) if reverse else state_dict.items()
    for key, tensor in items:
        if key_substring in key:
            return tensor
    raise ValueError(f"Could not find tensor with key containing '{key_substring}'")


def get_input_hw(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return tensor.shape[2], tensor.shape[3]


def get_input_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the input from the first convolutional layer's weights.
    """
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return normalize_dtype(tensor.dtype)


def get_output_dtype(state_dict: dict[str, torch.Tensor]) -> DataType:
    """
    Get the data type (dtype) of the output from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return normalize_dtype(tensor.dtype)


def get_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of input channels from the first convolutional layer's weights.
    """
    tensor = find_tensor_by_key(state_dict, "encoder._conv_stem.weight")
    return int(tensor.shape[1])


def get_output_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """
    Get number of output channels from the segmentation head's last conv layer.
    """
    tensor = find_tensor_by_key(state_dict, "segmentation_head.0.weight", reverse=True)
    return int(tensor.shape[0])


def extract_value_scaling(transforms: K.AugmentationSequential) -> list[dict[str, Any]]:
    children = list(transforms.children())

    def _tensor_to_value(tensor) -> Any:
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()

    scaling_defs = []

    for t in children:
        if isinstance(t, K.Normalize):
            buffers = dict(t.named_buffers()) if hasattr(t, "named_buffers") else {}
            mean = buffers.get("mean")
            stddev = buffers.get("std")

            if mean is None or stddev is None:
                flags = getattr(t, "flags", {})
                mean = mean or flags.get("mean")
                stddev = stddev or flags.get("std")

            if mean is None or stddev is None:
                raise AttributeError("Normalize transform missing mean/std info")

            scaling_defs.append(
                {
                    "type": "z-score",
                    "mean": int(_tensor_to_value(mean)),
                    "stddev": int(_tensor_to_value(stddev)),
                }
            )

        elif isinstance(t, K.AugmentationSequential):
            scaling_defs.extend(extract_value_scaling(t))

    return scaling_defs
