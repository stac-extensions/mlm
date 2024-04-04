from typing import Any, List, Literal, Optional, Set, TypeAlias, Union

from pydantic import BaseModel, Field

from stac_model.base import DataType, ProcessingExpression


class InputArray(BaseModel):
    shape: List[Union[int, float]] = Field(..., min_items=1)
    dim_order: List[str] = Field(..., min_items=1)
    data_type: DataType


class Statistics(BaseModel):
    minimum: Optional[List[Union[float, int]]] = None
    maximum: Optional[List[Union[float, int]]] = None
    mean: Optional[List[float]] = None
    stddev: Optional[List[float]] = None
    count: Optional[List[int]] = None
    valid_percent: Optional[List[float]] = None


class Band(BaseModel):
    name: str
    description: Optional[str] = None
    nodata: Union[float, int, str]
    data_type: str
    unit: Optional[str] = None


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


class ModelInput(BaseModel):
    name: str
    bands: List[str]
    input: InputArray
    norm_by_channel: bool = None
    norm_type: NormalizeType = None
    norm_clip: Optional[List[Union[float, int]]] = None
    resize_type: ResizeType = None
    statistics: Optional[Union[Statistics, List[Statistics]]] = None
    pre_processing_function: Optional[ProcessingExpression] = None
