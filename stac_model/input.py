from typing import Annotated, List, Literal, Optional, TypeAlias, Union

from pydantic import Field

from stac_model.base import DataType, MLMBaseModel, Number, OmitIfNone, ProcessingExpression


class InputStructure(MLMBaseModel):
    shape: List[Union[int, float]] = Field(min_items=1)
    dim_order: List[str] = Field(min_items=1)
    data_type: DataType


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
        "inf"
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


class ModelInput(MLMBaseModel):
    name: str
    bands: List[str]  # order is critical here (same index as dim shape), allow duplicate if the model needs it somehow
    input: InputStructure
    norm_by_channel: Annotated[Optional[bool], OmitIfNone] = None
    norm_type: Annotated[Optional[NormalizeType], OmitIfNone] = None
    norm_clip: Annotated[Optional[List[Union[float, int]]], OmitIfNone] = None
    resize_type: Annotated[Optional[ResizeType], OmitIfNone] = None
    statistics: Annotated[Optional[List[MLMStatistic]], OmitIfNone] = None
    pre_processing_function: Optional[ProcessingExpression] = None
