import json
from pathlib import Path
from collections.abc import Iterable
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
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
import zipfile

# Optional imports for ML frameworks
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import sklearn
    import joblib
except ImportError:
    sklearn = None
    joblib = None

try:
    import onnx
except ImportError:
    onnx = None


T = TypeVar(
    "T",
    pystac.Collection,
    pystac.Item,
    pystac.Asset,  # item_assets.AssetDefinition,
)

SchemaName = Literal["mlm"]
SCHEMA_URI: str = "https://stac-extensions.github.io/mlm/v1.4.0/schema.json"
PREFIX = f"{get_args(SchemaName)[0]}:"


def mlm_prefix_adder(field_name: str) -> str:
    return "mlm:" + field_name


class MLModelProperties(Runtime):
    name: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    tasks: set[ModelTask]
    input: list[ModelInput]
    output: list[ModelOutput]

    total_parameters: int
    pretrained: Annotated[bool | None, OmitIfNone] = Field(default=True)
    pretrained_source: Annotated[str | None, OmitIfNone] = None

    # Add framework and artifact_type if not already present from Runtime
    framework: Annotated[str | None, OmitIfNone] = None
    framework_version: Annotated[str | None, OmitIfNone] = None
    artifact_type: Annotated[str | None, OmitIfNone] = None # Added for consistency

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
    @classmethod
    def export_model(
        cls,
        model: Any,
        properties: MLModelProperties,
        output_path: Union[str, Path],
    ) -> tuple[str, str]:
        """
        Saves a model and its MLM properties to a specified path, returning
        the determined framework and artifact type.

        This method attempts to package the model artifact and its
        MLModelProperties (serialized as JSON) together. The saving
        strategy depends on the model's framework.

        Args:
            model: The machine learning model instance (PyTorch, TF/Keras, scikit-learn, ONNX).
            properties: The MLModelProperties describing the model.
            output_path: The file or directory path to save the packaged model and metadata.

        Returns:
            A tuple containing the determined framework (str) and artifact_type (str).

        Raises:
            ImportError: If the required ML framework library is not installed.
            TypeError: If the model type is not supported.
            Exception: For framework-specific saving errors.
        """
        output_path = Path(output_path)
        mlm_data = properties.model_dump_json(by_alias=True, indent=2)
        # Use framework from properties if provided, otherwise try to detect
        framework = properties.framework.lower() if properties.framework else None
        artifact_type = None
        detected_framework = None # Store detected framework separately

        # --- PyTorch ---
        if torch and isinstance(model, torch.nn.Module):
            detected_framework = "pytorch"
            if not framework:
                framework = detected_framework
            if framework != detected_framework:
                 raise ValueError(
                     f"Framework '{framework}' specified in MLModelProperties does not match detected framework '{detected_framework} of the model'."
                 )
            try:
                # ExportedProgram with all operators so the nn.Module can be accessed on load
                # https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html
                example_input = torch.randn(1, 3, 256, 256) # TODO get this from MLM properties
                ep_for_training = torch.export.export_for_training(model, (example_input,))
                torch.export.save(ep_for_training, output_path, extra_files={"mlm_properties": mlm_data})
                artifact_type = "torch.save"
                print(f"PyTorch model and MLM metadata saved to: {output_path}")
            except Exception as e:
                print(f"Error saving PyTorch model: {e}")
                raise

        else:
            supported_frameworks = []
            if torch:
                supported_frameworks.append("PyTorch")
            if tf:
                supported_frameworks.append("TensorFlow/Keras")
            if sklearn:
                supported_frameworks.append("Scikit-learn")
            if onnx:
                supported_frameworks.append("ONNX")
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                f"Supported frameworks with detected libraries: {', '.join(supported_frameworks)}"
            )

        return framework, artifact_type

    def load_model(self) -> Any:
        """
        Loads a model's framework representation from the MLM definition
        stored in the associated STAC object (Item/Asset).

        This method finds the asset with the 'mlm:model' role, reads its
        href and mlm:artifact_type, and attempts to load the model artifact.
        It may also load MLM properties if they are packaged with the model.

        Returns:
            The loaded model object (framework-specific).

        Raises:
            ValueError: If the model asset or required properties are missing.
            ImportError: If the required ML framework library is not installed.
            FileNotFoundError: If the model artifact href cannot be found.
            Exception: For framework-specific loading errors.
        """
        model_asset = None
        asset_href = None
        artifact_type = None
        mlm_props_dict = None # To store properties loaded from archive

        # Find the STAC object this extension is attached to
        stac_object = getattr(self, 'item', getattr(self, 'asset', getattr(self, 'collection', None)))
        if not stac_object:
             raise ValueError("MLModelExtension is not attached to a valid STAC object (Item, Asset, Collection).")

        # Find the asset dictionary
        assets_dict = None
        if hasattr(stac_object, 'assets'):
            assets_dict = stac_object.assets
        elif isinstance(stac_object, pystac.Asset): # If the extension is directly on an Asset
             # We need the owner (Item/Collection) to get the full context potentially
             owner = getattr(stac_object, 'owner', None)
             if owner and hasattr(owner, 'assets'):
                  # Find the asset key within the owner's assets
                  for key, asset_obj in owner.assets.items():
                       if asset_obj == stac_object:
                            model_asset = asset_obj # Found it
                            asset_href = model_asset.href
                            break
                  if not model_asset:
                       print("Warning: Extension is on an Asset, but couldn't find it in owner's assets.")
                       # Fallback: treat the asset's own fields
                       model_asset = stac_object
                       asset_href = model_asset.href
             else: # Asset has no owner or owner has no assets dict
                  model_asset = stac_object
                  asset_href = model_asset.href
        else:
             raise ValueError("Cannot find assets in the STAC object.")

        # Find the specific asset with the 'mlm:model' role if not already identified
        if not model_asset and assets_dict:
            for asset in assets_dict.values():
                # pystac Assets have .extra_fields for roles, not direct attribute
                if 'mlm:model' in asset.extra_fields.get('roles', []):
                    model_asset = asset
                    asset_href = model_asset.href
                    break

        if not model_asset:
            raise ValueError("Could not find an asset with the 'mlm:model' role.")

        if not asset_href:
            raise ValueError("Model asset found, but it does not have an 'href'.")

        # Get artifact type from the asset itself
        # pystac Assets store extension fields in .extra_fields
        artifact_type = model_asset.extra_fields.get('mlm:artifact_type')

        if not artifact_type:
             raise ValueError("Model asset must have the 'mlm:artifact_type' property defined in its extra_fields.")

        # Resolve the model path (handle relative paths)
        try:
            model_path = Path(pystac.utils.make_absolute_href(asset_href, start_href=stac_object.get_self_href(), start_is_dir=False))
            if not model_path.exists():
                 # Check if it's a directory path that might exist
                 if not (model_path.is_dir() and artifact_type == "tf.keras.Model.export"):
                      raise FileNotFoundError
        except Exception:
             raise FileNotFoundError(f"Model artifact not found or inaccessible at resolved href: {asset_href}")


        # --- Load based on artifact_type ---
        loaded_model = None

        # PyTorch (.pt file with state_dict and potentially mlm_properties)
        if artifact_type == "torch.save":
            if not torch:
                raise ImportError("PyTorch is required to load this model artifact, but it is not installed.")
            try:
                # Load using torch.export.load for ExportedProgram
                # https://pytorch.org/docs/stable/export.html#torch.export.load
                ep = torch.export.load(model_path)
                loaded_model = ep.module() # Access the original nn.Module

                # Attempt to load extra files (MLM properties) if they exist
                # Note: torch.export.load doesn't directly expose extra_files like torch.load
                # We might need to rely on the user accessing them separately if needed,
                # or consider if they should be part of the STAC metadata instead.
                # For now, we assume the essential model is loaded.
                # If mlm_properties were saved, they are inside the file but not directly accessible via torch.export.load API.

                print(f"PyTorch model loaded from: {model_path}")

            except Exception as e:
                print(f"Error loading PyTorch model from {model_path}: {e}")
                raise
        else:
            raise ValueError(f"Unsupported 'mlm:artifact_type' for loading: {artifact_type}")

        if not loaded_model:
             raise RuntimeError(f"Model loading failed for artifact type '{artifact_type}' from {model_path}")

        # Optionally attach loaded properties to the model (if we could load them)
        # if mlm_props_dict:
        #      setattr(loaded_model, '_mlm_properties', mlm_props_dict)

        return loaded_model

    @property
    def name(self) -> SchemaName:
        return cast(SchemaName, get_args(SchemaName)[0])

    def apply(
        self,
        properties: MLModelProperties | dict[str, Any],
    ) -> None:
        """
        Applies Machine Learning Model Extension properties to the extended :mod:`~pystac` object.
        """
        if isinstance(properties, dict):
            # Ensure framework and artifact_type are handled if present
            properties_obj = MLModelProperties(**properties)
        elif isinstance(properties, MLModelProperties):
             properties_obj = properties
        else:
             raise TypeError("properties must be MLModelProperties or dict")

        # Use model_dump to include potential aliases
        data_json = json.loads(properties_obj.model_dump_json(by_alias=True, exclude_none=True))
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
        obj: pystac.Collection | pystac.Item | pystac.Asset,  # item_assets.AssetDefinition
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
            pystac.ExtensionTypeError: If an invalid object type is passed.
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

    def get_mlm_property(self, prop: str) -> list[Any] | None:
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

    additional_read_properties: Iterable[dict[str, Any]] | None = None
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
