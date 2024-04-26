import json
from typing import (
    Annotated,
    Any,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)

import pystac
from pydantic import ConfigDict, Field
from pydantic.fields import FieldInfo
from pystac.extensions.base import (
    ExtensionManagementMixin,
    PropertiesExtension,
    SummariesExtension,
)

from stac_model.base import ModelTask, OmitIfNone
from stac_model.input import ModelInput
from stac_model.output import ModelOutput
from stac_model.runtime import Runtime

T = TypeVar(
    "T",
    pystac.Collection,
    pystac.Item,
    pystac.Asset,  # item_assets.AssetDefinition,
)

SchemaName = Literal["mlm"]
SCHEMA_URI: str = "https://crim-ca.github.io/mlm-extension/v1.1.0/schema.json"
PREFIX = f"{get_args(SchemaName)[0]}:"


def mlm_prefix_adder(field_name: str) -> str:
    return "mlm:" + field_name


class MLModelProperties(Runtime):
    name: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    tasks: Set[ModelTask]
    input: List[ModelInput]
    output: List[ModelOutput]

    total_parameters: int
    pretrained: Annotated[Optional[bool], OmitIfNone] = Field(default=True)
    pretrained_source: Annotated[Optional[str], OmitIfNone] = None

    model_config = ConfigDict(alias_generator=mlm_prefix_adder, populate_by_name=True, extra="ignore")


class MLModelExtension(
    Generic[T],
    PropertiesExtension,
    # FIXME: resolve typing incompatibility?
    #   'pystac.Asset' does not derive from STACObject
    #   therefore, it technically cannot be used in 'ExtensionManagementMixin[T]'
    #   however, this makes our extension definition much easier and avoids lots of code duplication
    ExtensionManagementMixin[  # type: ignore[type-var]
        Union[
            pystac.Collection,
            pystac.Item,
            pystac.Asset,
        ]
    ],
):
    @property
    def name(self) -> SchemaName:
        return cast(SchemaName, get_args(SchemaName)[0])

    def apply(
        self,
        properties: Union[MLModelProperties, dict[str, Any]],
    ) -> None:
        """
        Applies Machine Learning Model Extension properties to the extended :mod:`~pystac` object.
        """
        if isinstance(properties, dict):
            properties = MLModelProperties(**properties)
        data_json = json.loads(properties.model_dump_json(by_alias=True))
        for prop, val in data_json.items():
            self._set_property(prop, val)

    @classmethod
    def get_schema_uri(cls) -> str:
        return SCHEMA_URI

    @overload
    @classmethod
    def ext(cls, obj: pystac.Asset, add_if_missing: bool = False) -> "AssetMLModelExtension": ...

    @overload
    @classmethod
    def ext(cls, obj: pystac.Item, add_if_missing: bool = False) -> "ItemMLModelExtension": ...

    @overload
    @classmethod
    def ext(cls, obj: pystac.Collection, add_if_missing: bool = False) -> "CollectionMLModelExtension": ...

    # @overload
    # @classmethod
    # def ext(cls, obj: item_assets.AssetDefinition, add_if_missing: bool = False) -> "ItemAssetsMLModelExtension":
    #     ...

    @classmethod
    def ext(
        cls,
        obj: Union[pystac.Collection, pystac.Item, pystac.Asset],  # item_assets.AssetDefinition
        add_if_missing: bool = False,
    ) -> Union[
        "CollectionMLModelExtension",
        "ItemMLModelExtension",
        "AssetMLModelExtension",
    ]:
        """
        Extends the given STAC Object with properties from the :stac-ext:`Machine Learning Model Extension <mlm>`.

        This extension can be applied to instances of :class:`~pystac.Item` or :class:`~pystac.Asset`.

        Args:
            obj: STAC Object to extend with the MLM extension fields.
            add_if_missing: Add the MLM extension schema URI to the object if not already in `stac_extensions`.

        Returns:
            Extended object.

        Raises:
            pystac.ExtensionTypeError : If an invalid object type is passed.
        """
        if isinstance(obj, pystac.Collection):
            cls.ensure_has_extension(obj, add_if_missing)
            return CollectionMLModelExtension(obj)
        elif isinstance(obj, pystac.Item):
            cls.ensure_has_extension(obj, add_if_missing)
            return ItemMLModelExtension(obj)
        elif isinstance(obj, pystac.Asset):
            cls.ensure_owner_has_extension(obj, add_if_missing)
            return AssetMLModelExtension(obj)
        # elif isinstance(obj, item_assets.AssetDefinition):
        #     cls.ensure_owner_has_extension(obj, add_if_missing)
        #     return ItemAssetsMLModelExtension(obj)
        else:
            raise pystac.ExtensionTypeError(cls._ext_error_message(obj))

    @classmethod
    def summaries(cls, obj: pystac.Collection, add_if_missing: bool = False) -> "SummariesMLModelExtension":
        """Returns the extended summaries object for the given collection."""
        cls.ensure_has_extension(obj, add_if_missing)
        return SummariesMLModelExtension(obj)


class SummariesMLModelExtension(SummariesExtension):
    """
    Summaries annotated with the Machine Learning Model Extension.

    A concrete implementation of :class:`~SummariesExtension` that extends
    the ``summaries`` field of a :class:`~pystac.Collection` to include properties
    defined in the :stac-ext:`Machine Learning Model <mlm>`.
    """

    def _check_mlm_property(self, prop: str) -> FieldInfo:
        try:
            return MLModelProperties.model_fields[prop]
        except KeyError as err:
            raise AttributeError(f"Name '{prop}' is not a valid MLM property.") from err

    def _validate_mlm_property(self, prop: str, summaries: list[Any]) -> None:
        # ignore mypy issue when combined with Annotated
        #   - https://github.com/pydantic/pydantic/issues/6713
        #   - https://github.com/pydantic/pydantic/issues/5190
        model = MLModelProperties.model_construct()  # type: ignore[call-arg]
        validator = MLModelProperties.__pydantic_validator__
        for value in summaries:
            validator.validate_assignment(model, prop, value)

    def get_mlm_property(self, prop: str) -> Optional[list[Any]]:
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
    """
    Item annotated with the Machine Learning Model Extension.

    A concrete implementation of :class:`MLModelExtension` on an
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


# class ItemAssetsMLModelExtension(MLModelExtension[item_assets.AssetDefinition]):
#     properties: dict[str, Any]
#     asset_defn: item_assets.AssetDefinition
#
#     def __init__(self, item_asset: item_assets.AssetDefinition):
#         self.asset_defn = item_asset
#         self.properties = item_asset.properties


class AssetMLModelExtension(MLModelExtension[pystac.Asset]):
    """
    Asset annotated with the Machine Learning Model Extension.

    A concrete implementation of :class:`MLModelExtension` on an
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


# __all__ = [
#     "MLModelExtension",
#     "ModelInput",
#     "InputArray",
#     "Band",
#     "Statistics",
#     "ModelOutput",
#     "Asset",
#     "Runtime",
#     "Container",
#     "Asset",
# ]
