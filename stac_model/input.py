from typing import Annotated, Any, List, Literal, Optional, Sequence, TypeAlias, Union
from typing_extensions import Self

from pydantic import Field, model_validator

from stac_model.base import DataType, MLMBaseModel, Number, OmitIfNone, ProcessingExpression


class InputStructure(MLMBaseModel):
    shape: List[Union[int, float]] = Field(min_items=1)
    dim_order: List[str] = Field(min_items=1)
    data_type: DataType

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if len(self.shape) != len(self.dim_order):
            raise ValueError("Dimension order and shape must be of equal length for corresponding indices.")
        return self


class MLMStatistic(MLMBaseModel):  # FIXME: add 'Statistics' dep from raster extension (cases required to be triggered)
    minimum: Annotated[Optional[Number], OmitIfNone] = None
    maximum: Annotated[Optional[Number], OmitIfNone] = None
    mean: Annotated[Optional[Number], OmitIfNone] = None
    stddev: Annotated[Optional[Number], OmitIfNone] = None
    count: Annotated[Optional[int], OmitIfNone] = None
    valid_percent: Annotated[Optional[Number], OmitIfNone] = None


NormalizeType: TypeAlias = Optional[
    Literal[
        "min-max",
        "z-score",
        "l1",
        "l2",
        "l2sqr",
        "hamming",
        "hamming2",
        "type-mask",
        "relative",
        "inf",
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
    norm_by_channel: Annotated[Optional[bool], OmitIfNone] = None
    norm_type: Annotated[Optional[NormalizeType], OmitIfNone] = None
    norm_clip: Annotated[Optional[List[Union[float, int]]], OmitIfNone] = None
    resize_type: Annotated[Optional[ResizeType], OmitIfNone] = None
    statistics: Annotated[Optional[List[MLMStatistic]], OmitIfNone] = None
    pre_processing_function: Optional[ProcessingExpression] = None
