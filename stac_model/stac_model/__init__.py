"""A PydanticV2 validation and serialization library for the STAC ML Model Extension"""

from importlib import metadata

try:
    __version__ = metadata.version("stac-model")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
