from typing import List, Literal, Union

from pydantic import (
    BaseModel,
)


class Geometry(BaseModel):
    type: str
    coordinates: List


class GeoJSONPoint(Geometry):
    type: Literal["Point"]
    coordinates: List[float]


class GeoJSONMultiPoint(Geometry):
    type: Literal["MultiPoint"]
    coordinates: List[List[float]]


class GeoJSONPolygon(Geometry):
    type: Literal["Polygon"]
    coordinates: List[List[List[float]]]


class GeoJSONMultiPolygon(Geometry):
    type: Literal["MultiPolygon"]
    coordinates: List[List[List[List[float]]]]


AnyGeometry = Union[
    Geometry,
    GeoJSONPoint,
    GeoJSONMultiPoint,
    GeoJSONPolygon,
    GeoJSONMultiPolygon,
]
