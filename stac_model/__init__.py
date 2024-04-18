"""
A PydanticV2/PySTAC validation and serialization library for the STAC Machine Learning Model Extension.
"""

from importlib import metadata

try:
    __version__ = metadata.version("stac-model")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
