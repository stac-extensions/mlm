from typing import List, Optional, Literal, Dict, Literal
from pydantic import (
    BaseModel,
    Field,
    AnyUrl
)
class Array(BaseModel):
    shape: Optional[List[int]]
    dim_order: Literal["bhw", "bchw", "bthw", "btchw"]
    dtype: str = Field(..., regex="^(uint8|uint16|int16|int32|float16|float32|float64)$")

class Statistics(BaseModel):
    minimum: List[float | int]
    maximum: List[float | int]
    mean: List[float]
    stddev: List[float]
    count: List[int]
    valid_percent: List[float]

class Band(BaseModel):
    name: str
    description: str
    nodata: float | int | str
    data_type: str
    unit: Optional[str]

class ModelInput(BaseModel):
    name: str
    bands: List[Band]
    input_array: Array
    params: Optional[
        Dict[str, int | float | str]
    ] = None
    scaling_factor: float
    norm_by_channel: bool
    norm_type: Literal["min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "none"]
    rescale_type: Literal["crop", "pad", "interpolation", "none"]
    statistics: Optional[Statistics]
    pre_processing_function: str | AnyUrl
