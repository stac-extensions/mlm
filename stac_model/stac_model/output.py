from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from enum import Enum
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

class ResultArray(BaseModel):
    shape: List[Union[int,float]]
    dim_names: List[str]
    dtype: str = Field(..., pattern="^(uint8|uint16|int16|int32|float16|float32|float64)$")

class ModelOutput(BaseModel):
    task: TaskEnum
    number_of_classes: int = None
    result_array: ResultArray = None
    class_name_mapping: Optional[Dict[str, int]] = None
    post_processing_function: Optional[str] = None
