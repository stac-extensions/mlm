from pydantic import BaseModel, Field, AnyUrl
from typing import List, Optional, Literal, Any,  List, Tuple, Dict, Optional, Literal
from pydantic import (
    BaseModel,
    Field,
    AnyUrl
)
from enum import Enum
from .runtime import ModelArtifact

class Band(BaseModel):
    name: str
    description: str
    nodata: float | int | str
    data_type: str
    unit: Optional[str]

class TensorObject(BaseModel):
    batch: int = Field(..., gt=0)
    time: Optional[int] = Field(..., gt=0)
    channels: Optional[int] = Field(..., gt=0)
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    dim_order: Literal["bhw", "bchw", "bthw", "btchw", "bcthw"]

class Statistics(BaseModel):
    minimum: List[float | int]
    maximum: List[float | int]
    mean: List[float]
    stddev: List[float]
    count: List[int]
    valid_percent: List[float]

class ModelInput(BaseModel):
    name: str
    band_names: List[str]
    input_tensors: TensorObject
    params: Optional[
        Dict[str, int | float | str]
    ] = None
    scaling_factor: float
    norm_by_channel: str
    norm_type: Literal["min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "none"]
    rescale_type: Literal["crop", "pad", "interpolation", "none"]
    statistics: Optional[Statistics]
    pre_processing_function: str | AnyUrl

class ArchitectureObject(BaseModel):
    total_parameters: int
    on_disk_size_mb: float
    ram_size_mb: float
    model_type: str
    summary: str
    pretrained: str


class DockerObject(BaseModel):
    docker_file: str
    image_name: str
    tag: str
    working_dir: str
    run: str
    accelerator: bool

class RuntimeObject(BaseModel):
    framework: str
    version: str
    model_artifact: ModelArtifact
    model_handler: str
    model_src_url: str
    model_commit_hash: str
    docker: List[DockerObject]
    batch_size_suggestion: int
    hardware_suggestion: str | AnyUrl

class TaskEnum(str, Enum):
    regression = "regression"
    classification = "classification"
    object_detection = "object detection"
    semantic_segmentation = "semantic segmentation"
    instance_segmentation  = "instance segmentation"
    panoptic_segmentation = "panoptic segmentation"
    multi_modal = "multi-modal"
    similarity_search = "similarity search"
    image_captioning = "image captioning"
    generative =  "generative"

class ClassMap(BaseModel):
    class_to_label_id: Dict[str, int]

    # Property to reverse the mapping
    @property
    def label_id_to_class(self) -> Dict[int, str]:
        # Reverse the mapping
        return {v: k for k, v in self.class_to_label_id.items()}

    def get_class(self, class_id: int) -> str:
        """Get class name from class id."""
        if class_id not in self.label_id_to_class:
            raise ValueError(f"Class ID '{class_id}' not found")
        return self.label_id_to_class[class_id]

    def get_label_id(self, class_name: str) -> int:
        """Get class id from class name."""
        if class_name not in self.class_to_label_id:
            raise ValueError(f"Class name '{class_name}' not found")
        return self.class_to_label_id[class_name]

class OutputObject(BaseModel):
    task: TaskEnum
    number_of_classes: int
    final_layer_size: List[int]
    class_name_mapping: ClassMap
    post_processing_function: str

class DeepLearningModelExtension(BaseModel):
    bands: List[Band]
    dlm_input: ModelInput
    dlm_architecture: ArchitectureObject
    dlm_runtime: RuntimeObject
    dlm_output: OutputObject
