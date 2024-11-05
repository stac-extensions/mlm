from typing import Annotated, Any, List, Literal, Optional, Sequence, TypeAlias, Union
from typing_extensions import Self

from pydantic import Field, model_validator

from stac_model.base import DataType, MLMBaseModel, Number, OmitIfNone, ProcessingExpression


class InputStructure(MLMBaseModel):
    shape: List[Union[int, float]] = Field(min_length=1)
    dim_order: List[str] = Field(min_length=1)
    data_type: DataType

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if len(self.shape) != len(self.dim_order):
            raise ValueError("Dimension order and shape must be of equal length for corresponding indices.")
        return self


class ValueScalingClipMin(MLMBaseModel):
    type: Literal["clip-min"] = "clip-min"
    minimum: Number


class ValueScalingClipMax(MLMBaseModel):
    type: Literal["clip-max"] = "clip-max"
    maximum: Number


class ValueScalingClip(MLMBaseModel):
    type: Literal["clip"] = "clip"
    minimum: Number
    maximum: Number


class ValueScalingMinMax(MLMBaseModel):
    type: Literal["min-max"] = "min-max"
    minimum: Number
    maximum: Number


class ValueScalingZScore(MLMBaseModel):
    type: Literal["z-score"] = "z-score"
    mean: Number
    stddev: Number


class ValueScalingOffset(MLMBaseModel):
    type: Literal["offset"] = "offset"
    value: Number


class ValueScalingScale(MLMBaseModel):
    type: Literal["scale"] = "scale"
    value: Number


class ValueScalingProcessingExpression(ProcessingExpression):
    type: Literal["processing"] = "processing"


ValueScalingObject: TypeAlias = Optional[
    Union[
        ValueScalingMinMax,
        ValueScalingZScore,
        ValueScalingClip,
        ValueScalingClipMin,
        ValueScalingClipMax,
        ValueScalingOffset,
        ValueScalingScale,
        ValueScalingProcessingExpression,
    ]
]

ResizeType: TypeAlias = Optional[
    Literal[
        "crop",
        "pad",
        "interpolation-nearest",
        "interpolation-linear",
        "interpolation-cubic",
        "interpolation-area",
        "interpolation-lanczos4",
        "interpolation-max",
        "wrap-fill-outliers",
        "wrap-inverse-map",
    ]
]


class ModelBand(MLMBaseModel):
    name: str = Field(
        description=(
            "Name of the band to use for the input, "
            "referring to the name of an entry in a 'bands' definition from another STAC extension."
        )
    )
    # similar to 'ProcessingExpression', but they can be omitted here
    format: Annotated[Optional[str], OmitIfNone] = Field(
        default=None,
        description="",
    )
    expression: Annotated[Optional[Any], OmitIfNone] = Field(
        default=None,
        description="",
    )

    @model_validator(mode="after")
    def validate_expression(self) -> Self:
        if (  # mutually dependant
            (self.format is not None or self.expression is not None)
            and (self.format is None or self.expression is None)
        ):
            raise ValueError("Model band 'format' and 'expression' are mutually dependant.")
        return self


class ModelInput(MLMBaseModel):
    name: str
    # order is critical here (same index as dim shape), allow duplicate if the model needs it somehow
    bands: Sequence[Union[str, ModelBand]] = Field(
        description=(
            "List of bands that compose the input. "
            "If a string is used, it is implied to correspond to a named-band. "
            "If no band is needed for the input, use an empty array."
        ),
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
    input: InputStructure
    value_scaling: Annotated[Optional[List[ValueScalingObject]], OmitIfNone] = None
    resize_type: Annotated[Optional[ResizeType], OmitIfNone] = None
    pre_processing_function: Optional[ProcessingExpression] = None
