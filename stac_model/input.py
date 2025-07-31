from typing import Annotated, Literal, TypeAlias, Union
from typing_extensions import Self

from pydantic import Field, model_validator

from stac_model.base import (
    DataType,
    MLMBaseModel,
    ModelBandsOrVariablesReferences,
    Number,
    OmitIfNone,
    ProcessingExpression,
)


class InputStructure(MLMBaseModel):
    shape: list[Union[int, float]] = Field(min_length=1)
    dim_order: list[str] = Field(min_length=1)
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


ValueScalingObject: TypeAlias = Union[ValueScalingMinMax, ValueScalingZScore, ValueScalingClip, ValueScalingClipMin, ValueScalingClipMax, ValueScalingOffset, ValueScalingScale, ValueScalingProcessingExpression] | None

ResizeType: TypeAlias = Literal["crop", "pad", "interpolation-nearest", "interpolation-linear", "interpolation-cubic", "interpolation-area", "interpolation-lanczos4", "interpolation-max", "wrap-fill-outliers", "wrap-inverse-map"] | None


class ModelInput(ModelBandsOrVariablesReferences):
    name: str
    input: InputStructure
    value_scaling: Annotated[list[ValueScalingObject] | None, OmitIfNone] = None
    resize_type: Annotated[ResizeType | None, OmitIfNone] = None
    pre_processing_function: ProcessingExpression | list[ProcessingExpression] | None = None
