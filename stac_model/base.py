from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, model_serializer

Number: TypeAlias = Union[int, float]
JSON: TypeAlias = Union[
    Dict[str, "JSON"],
    List["JSON"],
    Number,
    bool,
    str,
    None,
]


@dataclass
class _OmitIfNone:
    pass


OmitIfNone = _OmitIfNone()


class MLMBaseModel(BaseModel):
    """
    Allows wrapping any field with an annotation to drop it entirely if unset.

    ```python
    field: Annotated[Optional[<desiredType>], OmitIfNone] = None
    # or
    field: Annotated[Optional[<desiredType>], OmitIfNone] = Field(default=None)
    ```

    Since `OmitIfNone` implies that the value could be `None` (even though it would be dropped),
    the `Optional` annotation must be specified to corresponding typings to avoid `mypy` lint issues.

    It is important to use `MLMBaseModel`, otherwise the serializer will not be called and applied.

    Reference: https://github.com/pydantic/pydantic/discussions/5461#discussioncomment-7503283
    """

    @model_serializer
    def model_serialize(self):
        omit_if_none_fields = {
            key: field
            for key, field in self.model_fields.items()
            if any(isinstance(m, _OmitIfNone) for m in field.metadata)
        }
        values = {
            self.__fields__[key].alias or key: val  # use the alias if specified
            for key, val in self
            if key not in omit_if_none_fields or val is not None
        }
        return values

    model_config = ConfigDict(
        populate_by_name=True,
    )


DataType: TypeAlias = Literal[
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "cint16",
    "cint32",
    "cfloat32",
    "cfloat64",
    "other",
]


class TaskEnum(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SCENE_CLASSIFICATION = "scene-classification"
    DETECTION = "detection"
    OBJECT_DETECTION = "object-detection"
    SEGMENTATION = "segmentation"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    INSTANCE_SEGMENTATION = "instance-segmentation"
    PANOPTIC_SEGMENTATION = "panoptic-segmentation"
    SIMILARITY_SEARCH = "similarity-search"
    GENERATIVE = "generative"
    IMAGE_CAPTIONING = "image-captioning"
    SUPER_RESOLUTION = "super-resolution"


ModelTaskNames: TypeAlias = Literal[
    "regression",
    "classification",
    "scene-classification",
    "detection",
    "object-detection",
    "segmentation",
    "semantic-segmentation",
    "instance-segmentation",
    "panoptic-segmentation",
    "similarity-search",
    "generative",
    "image-captioning",
    "super-resolution",
]


ModelTask = Union[ModelTaskNames, TaskEnum]


class ProcessingExpression(BaseModel):
    # FIXME: should use 'pystac' reference, but 'processing' extension is not implemented yet!
    format: str
    expression: Any
