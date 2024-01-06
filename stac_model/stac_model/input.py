from typing import List, Optional, Literal, Dict, Literal, Union
from pydantic import (
    BaseModel,
    Field,
    AnyUrl
)
class InputArray(BaseModel):
    shape: List[Union[int,float]]
    dim_order: Literal["bhw", "bchw", "bthw", "btchw"]
    dtype: str = Field(..., pattern="^(uint8|uint16|int16|int32|float16|float32|float64)$")

class Statistics(BaseModel):
    minimum: Optional[List[Union[float, int]]] = None
    maximum: Optional[List[Union[float, int]]] = None
    mean: Optional[List[float]] = None
    stddev: Optional[List[float]] = None
    count: Optional[List[int]] = None
    valid_percent: Optional[List[float]] = None

class Band(BaseModel):
    name: str
    description: str
    nodata: float | int | str
    data_type: str
    unit: Optional[str] = None

class Input(BaseModel):
    name: str
    bands: List[Band]
    input_array: InputArray
    norm_type: Literal["min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "none"]
    rescale_type: Literal["crop", "pad", "interpolation", "none"]
    norm_by_channel: bool
    params: Optional[
        Dict[str, int | float | str]
    ] = None
    scaling_factor: Optional[float] = None
    statistics: Optional[Statistics] = None
    pre_processing_function: Optional[str | AnyUrl] = None
