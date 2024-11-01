from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeAlias
from typing_extensions import Self

from pydantic import Field, model_validator

from stac_model.base import DataType, MLMBaseModel, Number, OmitIfNone, ProcessingExpression


class InputStructure(MLMBaseModel):
    shape: list[int | float] = Field(min_items=1)
    dim_order: list[str] = Field(min_items=1)
    data_type: DataType

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if len(self.shape) != len(self.dim_order):
            raise ValueError("Dimension order and shape must be of equal length for corresponding indices.")
        return self


class MLMStatistic(MLMBaseModel):  # FIXME: add 'Statistics' dep from raster extension (cases required to be triggered)
    minimum: Annotated[Number | None, OmitIfNone] = None
    maximum: Annotated[Number | None, OmitIfNone] = None
    mean: Annotated[Number | None, OmitIfNone] = None
    stddev: Annotated[Number | None, OmitIfNone] = None
    count: Annotated[int | None, OmitIfNone] = None
    valid_percent: Annotated[Number | None, OmitIfNone] = None


NormalizeType: TypeAlias = Literal["min-max", "z-score", "l1", "l2", "l2sqr", "hamming", "hamming2", "type-mask", "relative", "inf"] | None

ResizeType: TypeAlias = Literal["crop", "pad", "interpolation-nearest", "interpolation-linear", "interpolation-cubic", "interpolation-area", "interpolation-lanczos4", "interpolation-max", "wrap-fill-outliers", "wrap-inverse-map"] | None


class ModelBand(MLMBaseModel):
    name: str = Field(
        description=(
            "Name of the band to use for the input, "
            "referring to the name of an entry in a 'bands' definition from another STAC extension."
        )
    )
    # similar to 'ProcessingExpression', but they can be omitted here
    format: Annotated[str | None, OmitIfNone] = Field(
        default=None,
        description="",
    )
    expression: Annotated[Any | None, OmitIfNone] = Field(
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
    bands: Sequence[str | ModelBand] = Field(
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
    norm_by_channel: Annotated[bool | None, OmitIfNone] = None
    norm_type: Annotated[NormalizeType | None, OmitIfNone] = None
    norm_clip: Annotated[list[float | int] | None, OmitIfNone] = None
    resize_type: Annotated[ResizeType | None, OmitIfNone] = None
    statistics: Annotated[list[MLMStatistic] | None, OmitIfNone] = None
    pre_processing_function: ProcessingExpression | None = None
