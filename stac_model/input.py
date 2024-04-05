from typing import Any, Annotated, List, Literal, Optional, Set, TypeAlias, Union

from pystac.extensions.raster import Statistics
from pydantic import ConfigDict, Field, model_serializer

from stac_model.base import DataType, MLMBaseModel, ProcessingExpression, OmitIfNone

Number: TypeAlias = Union[int, float]


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


NormalizeType: TypeAlias = Optional[Literal[
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
]]

ResizeType: TypeAlias = Optional[Literal[
    "crop",
    "pad",
    "interpolation-nearest",
    "interpolation-linear",
    "interpolation-cubic",
    "interpolation-area",
    "interpolation-lanczos4",
    "interpolation-max",
    "wrap-fill-outliers",
    "wrap-inverse-map"
]]


class ModelInput(MLMBaseModel):
    name: str
    bands: List[str]  # order is critical here (same index as dim shape), allow duplicate if the model needs it somehow
    input: InputStructure
    norm_by_channel: Annotated[bool, OmitIfNone] = None
    norm_type: Annotated[NormalizeType, OmitIfNone] = None
    norm_clip: Annotated[List[Union[float, int]], OmitIfNone] = None
    resize_type: Annotated[ResizeType, OmitIfNone] = None
    statistics: Annotated[List[MLMStatistic], OmitIfNone] = None
    pre_processing_function: Optional[ProcessingExpression] = None
