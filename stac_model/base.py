from dataclasses import dataclass
from enum import Enum
from typing import Any, Annotated, Literal, Optional, Self, Sequence, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

Number: TypeAlias = int | float
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | Number | bool | str | None


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
        fields = getattr(self, "model_fields", self.__fields__)  # noqa
        values = {
            fields[key].alias or key: val  # use the alias if specified
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
    DOWNSCALING = "downscaling"


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
    "downscaling",
]


ModelTask = Union[ModelTaskNames, TaskEnum]


class ProcessingExpression(MLMBaseModel):
    """
    Expression used to perform a pre-processing or post-processing step on the input or output model data.
    """
    # FIXME: should use 'pystac' reference, but 'processing' extension is not implemented yet!
    format: str = Field(
        description="The type of the expression that is specified in the 'expression' property.",
    )
    expression: Any = Field(
        description=(
            "An expression compliant with the 'format' specified. "
            "The expression can be any data type and depends on the format given. "
            "This represents the processing operation to be applied on the entire data before or after the model."
        )
    )


class ModelCrossReferenceObject(MLMBaseModel):
    name: str = Field(
        description=(
            "Name of the reference to use for the input or output. "
            "The name must refer to an entry of a relevant STAC extension providing further definition details."
        )
    )
    # similar to 'ProcessingExpression', but they can be omitted here
    format: Annotated[Optional[str], OmitIfNone] = Field(
        default=None,
        description="The type of the expression that is specified in the 'expression' property.",
    )
    expression: Annotated[Optional[Any], OmitIfNone] = Field(
        default=None,
        description=(
            "An expression compliant with the 'format' specified. "
            "The expression can be any data type and depends on the format given. "
            "This represents the processing operation to be applied on the data before or after the model. "
            "Contrary to pre/post-processing expressions, this expression is applied only to the specific "
            "item it refers to."
        )
    )

    @model_validator(mode="after")
    def validate_expression(self) -> Self:
        if (  # mutually dependant
            (self.format is not None or self.expression is not None)
            and (self.format is None or self.expression is None)
        ):
            raise ValueError("Model band 'format' and 'expression' are mutually dependant.")
        return self


class ModelBand(ModelCrossReferenceObject):
    """
    Definition of a band reference in the model input or output.
    """


class ModelDataVariable(ModelCrossReferenceObject):
    """
    Definition of a data variable in the model input or output.
    """


class ModelBandsOrVariablesReferences(MLMBaseModel):
    bands: Annotated[Sequence[str | ModelBand] | None, OmitIfNone] = Field(
        description=(
            "List of bands that compose the data. "
            "If a string is used, it is implied to correspond to a named band. "
            "If no band is needed for the data, use an empty array, or omit the property entirely. "
            "If provided, order is critical to match the stacking method as aggregated 'bands' dimension "
            "in 'dim_order' and 'shape' lists."
        ),
        # default omission is interpreted the same as if empty list was provided, but populate it explicitly
        # if the user wishes to omit the property entirely, they can use `None` explicitly
        default=[],
        examples=[
            [
                "B01",
                {"name": "B02"},
                {
                    "name": "NDVI",
                    "format": "rio-calc",
                    "expression": "(B08 - B04) / (B08 + B04)",
                },
            ],
        ],
    )
    variables: Annotated[Sequence[str | ModelDataVariable] | None, OmitIfNone] = Field(
        description=(
            "List of variables that compose the data. "
            "If a string is used, it is implied to correspond to a named variable. "
            "If no variable is needed for the data, use an empty array, or omit the property entirely. "
            "If provided, order is critical to match the stacking method as aggregated 'variables' dimension "
            "in 'dim_order' and 'shape' lists."
        ),
        # default omission is interpreted the same as if empty list was provided, but populate it explicitly
        # if the user wishes to omit the property entirely, they can use `None` explicitly
        default=[],
        examples=[
            [
                "10m_u_component_of_wind",
                {"name": "10m_v_component_of_wind"},
                {
                    "name": "temperature_2m_celsius",
                    "format": "rio-calc",
                    "expression": "temperature_2m + 273.15",
                },
            ],
        ],
    )
