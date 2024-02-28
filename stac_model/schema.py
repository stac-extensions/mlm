import json
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
)

import pystac
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from pystac.extensions import item_assets
from pystac.extensions.base import (
    ExtensionManagementMixin,
    PropertiesExtension,
    S,  # generic pystac.STACObject
    SummariesExtension,
)

from .input import Band, InputArray, ModelInput, Statistics
from .output import ClassObject, ModelOutput, ResultArray, TaskEnum
from .runtime import Asset, Container, Runtime

T = TypeVar(
    "T", pystac.Collection, pystac.Item, pystac.Asset, item_assets.AssetDefinition
)

SchemaName = Literal["mlm"]
# TODO update
SCHEMA_URI: str = "https://raw.githubusercontent.com/crim-ca/dlm-extension/main/json-schema/schema.json"  # noqa: E501
PREFIX = f"{get_args(SchemaName)[0]}:"


def mlm_prefix_replacer(field_name: str) -> str:
    return field_name.replace("mlm_", "mlm:")


class MLModelProperties(BaseModel):
    mlm_name: str
    mlm_task: TaskEnum
    mlm_framework: str
    mlm_framework_version: str
    mlm_file_size: int
    mlm_memory_size: int
    mlm_input: List[ModelInput]
    mlm_output: List[ModelOutput]
    mlm_runtime: List[Runtime]
    mlm_total_parameters: int
    mlm_pretrained_source: str
    mlm_summary: str
    mlm_parameters: Optional[
        Dict[str, Union[int, str, bool, List[Union[int, str, bool]]]]
    ] = None  # noqa: E501

    model_config = ConfigDict(
        alias_generator=mlm_prefix_replacer, populate_by_name=True, extra="ignore"
    )


class MLModelExtension(
    Generic[T],
    PropertiesExtension,
    ExtensionManagementMixin[Union[pystac.Asset, pystac.Item, pystac.Collection]],
):
    @property
    def name(self) -> SchemaName:
        return get_args(SchemaName)[0]

    def apply(
        self,
        properties: Union[MLModelProperties, dict[str, Any]],
    ) -> None:
        """Applies Machine Learning Model Extension properties to the extended
        :class:`~pystac.Item` or :class:`~pystac.Asset`.
        """
        if isinstance(properties, dict):
            properties = MLModelProperties(**properties)
        data_json = json.loads(properties.model_dump_json(by_alias=True))
        for prop, val in data_json.items():
            self._set_property(prop, val)

    @classmethod
    def get_schema_uri(cls) -> str:
        return SCHEMA_URI

    @classmethod
    def has_extension(cls, obj: S):
        # FIXME: this override should be removed once an official and
        # versioned schema is released ignore the original implementation
        # logic for a version regex since in our case, the VERSION_REGEX
        # is not fulfilled (ie: using 'main' branch, no tag available...)
        ext_uri = cls.get_schema_uri()
        return obj.stac_extensions is not None and any(
            uri == ext_uri for uri in obj.stac_extensions
        )

    @classmethod
    def ext(cls, obj: T, add_if_missing: bool = False) -> "MLModelExtension[T]":
        """Extends the given STAC Object with properties from the
        :stac-ext:`Machine Learning Model Extension <mlm>`.

        This extension can be applied to instances of :class:`~pystac.Item` or
        :class:`~pystac.Asset`.

        Raises:

            pystac.ExtensionTypeError : If an invalid object type is passed.
        """
        if isinstance(obj, pystac.Collection):
            cls.ensure_has_extension(obj, add_if_missing)
            return cast(MLModelExtension[T], CollectionMLModelExtension(obj))
        elif isinstance(obj, pystac.Item):
            cls.ensure_has_extension(obj, add_if_missing)
            return cast(MLModelExtension[T], ItemMLModelExtension(obj))
        elif isinstance(obj, pystac.Asset):
            cls.ensure_owner_has_extension(obj, add_if_missing)
            return cast(MLModelExtension[T], AssetMLModelExtension(obj))
        elif isinstance(obj, item_assets.AssetDefinition):
            cls.ensure_owner_has_extension(obj, add_if_missing)
            return cast(MLModelExtension[T], ItemAssetsMLModelExtension(obj))
        else:
            raise pystac.ExtensionTypeError(cls._ext_error_message(obj))

    @classmethod
    def summaries(
        cls, obj: pystac.Collection, add_if_missing: bool = False
    ) -> "SummariesMLModelExtension":
        """Returns the extended summaries object for the given collection."""
        cls.ensure_has_extension(obj, add_if_missing)
        return SummariesMLModelExtension(obj)


class SummariesMLModelExtension(SummariesExtension):
    """A concrete implementation of :class:`~SummariesExtension` that extends
    the ``summaries`` field of a :class:`~pystac.Collection` to include properties
    defined in the :stac-ext:`Machine Learning Model <mlm>`.
    """

    def _check_mlm_property(self, prop: str) -> FieldInfo:
        try:
            return MLModelProperties.model_fields[prop]
        except KeyError as err:
            raise AttributeError(f"Name '{prop}' is not a valid MLM property.") from err

    def _validate_mlm_property(self, prop: str, summaries: list[Any]) -> None:
        model = MLModelProperties.model_construct()
        validator = MLModelProperties.__pydantic_validator__
        for value in summaries:
            validator.validate_assignment(model, prop, value)

    def get_mlm_property(self, prop: str) -> list[Any]:
        self._check_mlm_property(prop)
        return self.summaries.get_list(prop)

    def set_mlm_property(self, prop: str, summaries: list[Any]) -> None:
        self._check_mlm_property(prop)
        self._validate_mlm_property(prop, summaries)
        self._set_summary(prop, summaries)

    def __getattr__(self, prop):
        return self.get_mlm_property(prop)

    def __setattr__(self, prop, value):
        self.set_mlm_property(prop, value)


class ItemMLModelExtension(MLModelExtension[pystac.Item]):
    """A concrete implementation of :class:`MLModelExtension` on an
    :class:`~pystac.Item` that extends the properties of the Item to
    include properties defined in the :stac-ext:`Machine Learning Model
    Extension <mlm>`.

    This class should generally not be instantiated directly. Instead, call
    :meth:`MLModelExtension.ext` on an :class:`~pystac.Item` to extend it.
    """

    def __init__(self, item: pystac.Item):
        self.item = item
        self.properties = item.properties

    def __repr__(self) -> str:
        return f"<ItemMLModelExtension Item id={self.item.id}>"


class ItemAssetsMLModelExtension(MLModelExtension[item_assets.AssetDefinition]):
    properties: dict[str, Any]
    asset_defn: item_assets.AssetDefinition

    def __init__(self, item_asset: item_assets.AssetDefinition):
        self.asset_defn = item_asset
        self.properties = item_asset.properties


class AssetMLModelExtension(MLModelExtension[pystac.Asset]):
    """A concrete implementation of :class:`MLModelExtension` on an
    :class:`~pystac.Asset` that extends the Asset fields to include
    properties defined in the :stac-ext:`Machine Learning Model
    Extension <mlm>`.

    This class should generally not be instantiated directly. Instead, call
    :meth:`MLModelExtension.ext` on an :class:`~pystac.Asset` to extend it.
    """

    asset_href: str
    """The ``href`` value of the :class:`~pystac.Asset` being extended."""

    properties: dict[str, Any]
    """The :class:`~pystac.Asset` fields, including extension properties."""

    additional_read_properties: Optional[Iterable[dict[str, Any]]] = None
    """If present, this will be a list containing 1 dictionary representing the
    properties of the owning :class:`~pystac.Item`."""

    def __init__(self, asset: pystac.Asset):
        self.asset_href = asset.href
        self.properties = asset.extra_fields
        if asset.owner and isinstance(asset.owner, pystac.Item):
            self.additional_read_properties = [asset.owner.properties]

    def __repr__(self) -> str:
        return f"<AssetMLModelExtension Asset href={self.asset_href}>"


class CollectionMLModelExtension(MLModelExtension[pystac.Collection]):
    def __init__(self, collection: pystac.Collection):
        self.collection = collection


__all__ = [
    "MLModelExtension",
    "ModelInput",
    "InputArray",
    "Band",
    "Statistics",
    "ModelOutput",
    "ClassObject",
    "Asset",
    "ResultArray",
    "Runtime",
    "Container",
    "Asset",
]
