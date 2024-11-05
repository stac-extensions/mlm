# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/stac-extensions/mlm/tree/main)

### Added
- Add explicit check of `value_scaling` sub-fields `minimum`, `maximum`, `mean`, `stddev`, etc. for
  corresponding `type` values `min-max` and `z-score` that depend on it.
- Allow different `value_scaling` operations per band/channel/dimension as needed by the model.
- Allow a `processing:expression` for a band/channel/dimension-specific `value_scaling` operation,
  granting more flexibility in the definition of input preparation in contrast to having it applied
  for the entire input (but still possible).

### Changed
- Moved `norm_type` to `value_scaling` object to better reflect the expected operation, which could be another
  operation than what is typically known as "normalization" or "standardization" techniques in machine learning.
- Moved `statistics` to `value_scaling` object to better reflect their mutual `type` and additional
  properties dependencies.

### Deprecated
- n/a

### Removed
- Removed `norm_type` enum values that were ambiguous regarding their expected result.
  Instead, a `processing:expression` should be employed to explicitly define the calculation they represent.
- Removed `norm_clip` property. It is now represented under `value_scaling` objects with a
  corresponding `type` definition.
- Removed `norm_by_channel` from `mlm:input` objects. If rescaling (previously normalization in the documentation)
  is a single value, broadcasting to the relevant bands should be performed implicitly.
  Otherwise, the amount of `value_scaling` objects should match the number of bands or channels involved in the input.

### Fixed
- Fix missing `mlm:artifact_type` property check for a Model Asset definition
  (fixes <https://github.com/stac-extensions/mlm/issues/42>).
  The `mlm:artifact_type` is now mutually and exclusively required by the corresponding Asset with `mlm:model` role.
- Fix check of disallowed unknown/undefined `mlm:`-prefixed fields
  (fixes [#41](https://github.com/stac-extensions/mlm/issues/41)).

## [v1.3.0](https://github.com/stac-extensions/mlm/tree/v1.3.0)

### Added
- Add `raster:bands` required property `name` for describing `mlm:input` bands
  (see [README - Bands and Statistics](README.md#bands-and-statistics) for details).
- Add README warnings about new extension `eo` and `raster` versions.

### Changed
- Split `ModelBands` and `AnyBandsRef` definitions in the JSON schema to allow them to be referenced individually.
- Move `AnyBandsRef` definition explicitly to STAC Item JSON schema, rather than implicitly inferred via `mlm:input`.
- Modified the JSON schema to use a `if` check of the `type` (STAC Item or Collection) prior to validating further
  properties. This allows some validators (e.g. `pystac`) to better report the *real* error that causes the schema
  to fail, rather than reporting the first mismatching `type` case with a poor error description to debug the issue.

### Deprecated
- n/a

### Removed
- Removed `$comment` entries from the JSON schema that are considered as invalid by some parsers.
- When `mlm:input` objects do **NOT** define band references (i.e.: `bands: []` is used), the JSON schema will not
  fail if an Asset with the `mlm:model` role contains a band definition. This is to allow MLM model definitions to
  simultaneously use some inputs with `bands` reference names while others do not.

### Fixed
- Band checks against [`eo`](https://github.com/stac-extensions/eo), [`raster`](https://github.com/stac-extensions/eo)
  or STAC Core 1.1 [`bands`](https://github.com/radiantearth/stac-spec/blob/master/commons/common-metadata.md#bands)
  when a `mlm:input` references names in `bands` are now properly validated.
- Fix the examples using `raster:bands` incorrectly defined in STAC Item properties.
  The correct use is for them to be defined under the STAC Asset using the `mlm:model` role.
- Fix the [EuroSAT ResNet pydantic example](stac_model/examples.py) that incorrectly referenced some `bands`
  in its `mlm:input` definition without providing any definition of those bands. The `eo:bands` properties have
  been added to the corresponding `model` Asset using
  the [`pystac.extensions.eo`](https://github.com/stac-utils/pystac/blob/main/pystac/extensions/eo.py) utilities.
- Fix various STAC Asset definitions erroneously employing `mlm:model` role instead of the intended `mlm:source_code`.

## [v1.2.0](https://github.com/stac-extensions/mlm/tree/v1.2.0)

### Added
- Add the missing JSON schema `item_assets` definition under a Collection to ensure compatibility with
  the [Item Assets](https://github.com/stac-extensions/item-assets) extension, as mentioned this specification.
- Add `ModelBand` representation using `name`, `format` and `expression` properties to allow derived band references
  (fixes [crim-ca/mlm-extension#7](https://github.com/stac-extensions/mlm/discussions/7)).

### Changed
- Adds a job to `.github/workflows/publish.yaml` to publish the `stac-model` package to PyPI.

### Deprecated
- n/a

### Removed
- Field `mlm:name` requirement to be unique. There is no way to guarantee this from a single Item's definition
  and their JSON schema validation. For uniqueness requirement, users should instead rely on the `id` property
  of the Item, which is ensured to be unique under the corresponding Collection, since it would not be retrievable
  otherwise (i.e.: `collections/{collectionID}/items/{itemID}`).

### Fixed
- Fix the validation strategy of the `mlm:model` role required by at least one Asset under a STAC Item.
  Although the role requirement was validated, the definition did not allow for other Assets without it to exist.
- Correct `stac-model` version in code and publish matching release on PyPI.

## [v1.1.0](https://github.com/stac-extensions/mlm/tree/v1.1.0)

### Added
- Add pattern for `mlm:framework`, needing at least one alphanumeric character,
  without leading or trailing non-alphanumeric characters.
- Add [`examples/item_eo_and_raster_bands.json`](examples/item_eo_and_raster_bands.json) demonstrating the original
  use case represented by the previous [`examples/item_eo_bands.json`](examples/item_eo_bands.json) contents.
- Add a `description` field for `mlm:input` and `mlm:output` definitions.

### Changed
- Adjust `scikit-learn` and `Hugging Face` framework names to match the format employed by the official documentation.

### Deprecated
- n/a

### Removed
- Removed combination of `mlm:input` with `bands: null` that could never occur due to pre-requirement of `type: array`.

### Fixed
- Fix `AnyBands` definition and use in the JSON schema to better consider possible use cases with `eo` extension.
- Fix [`examples/item_eo_bands.json`](examples/item_eo_bands.json) that was incorrectly also using `raster` extension.
  This is not fundamentally wrong, but it did not allow to validate the `eo` extension use case properly, since
  the `raster:bands` reference caused a bypass for the `mlm:input[*].bands` to succeed validation.

## [v1.0.0](https://github.com/stac-extensions/mlm/tree/v1.0.0)

### Added
- more [Task Enum](README.md#task-enum) tasks
- [Model Output Object](README.md#model-output-object)
- batch_size and hardware summary
- [`mlm:accelerator`, `mlm:accelerator_constrained`, `mlm:accelerator_summary`](README.md#accelerator-type-enum)
  to specify hardware requirements for the model
- Use common metadata
  [Asset Object](https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object)
  to refer to model asset and source code.
- use `classification:classes` in Model Output
- add `scene-classification` to the Enum Tasks to allow disambiguation between pixel-wise and patch-based classification

### Changed
- `disk_size` replaced by `file:size` (see [Best Practices - File Extension](best-practices.md#file-extension))
- `memory_size` under `dlm:architecture` moved directly under Item properties as `mlm:memory_size`
- replaced all hardware/accelerator/runtime definitions into distinct `mlm` fields directly under the
  STAC Item properties (top-level, not nested) to allow better search support by STAC API.
- reorganized `dlm:architecture` nested fields to exist at the top level of properties as `mlm:name`, `mlm:summary`
  and so on to provide STAC API search capabilities.
- replaced `normalization:mean`, etc. with [statistics](README.md#bands-and-statistics) from STAC 1.1 common metadata
- added `pydantic` models for internal schema objects in `stac_model` package and published to PYPI
- specified [rel_type](README.md#relation-types) to be `derived_from` and
  specify how model item or collection json should be named
- replaced all Enum Tasks names to use hyphens instead of spaces
- replaced `dlm:task` by `mlm:tasks` using an array of value instead of a single one, allowing models to represent
  multiple tasks they support simultaneously or interchangeably depending on context
- replace `pre_processing_function` and `post_processing_function` to use similar definitions
  to the [Processing Extension - Expression Object](https://github.com/stac-extensions/processing#expression-object)
  such that more extended definitions of custom processors can be defined.
- updated JSON schema to reflect changes of MLM fields

### Deprecated
- any `dlm`-prefixed field or property

### Removed
- Data Object, replaced with [Model Input Object](README.md#model-input-object) that uses the `name` field from
  the [common metadata band object][stac-bands] which also records `data_type` and `nodata` type

### Fixed
- n/a

[stac-bands]: https://github.com/radiantearth/stac-spec/blob/f9b3c59ba810541c9da70c5f8d39635f8cba7bcd/item-spec/common-metadata.md#bands

## [v1.0.0-beta3](https://github.com/crim-ca/dlm-extension/tree/v1.0.0-beta3)

### Added
- Added example model architecture summary text.

### Changed
- Modified `$id` if the extension schema to refer to the expected location when eventually released
  (`https://schemas.stacspec.org/v1.0.0-beta.3/extensions/dl-model/json-schema/schema.json`).
- Replaced `dtype` field by `data_type` to better align with the corresponding field of
  [`raster:bands`][raster-band-object].
- Replaced `nodata_value` field by `nodata` to better align with the corresponding field of
  [`raster:bands`][raster-band-object].
- Refactored schema to use distinct definitions and references instead of embedding all objects
  within `dl-model` properties.
- Allow schema to contain other `dlm:`-prefixed elements using `patternProperties` and explicitly
  deny other `additionalProperties`.
- Allow `class_name_mapping` to be directly provided as a mapping of index-based properties and class-name values.

[raster-band-object]: https://github.com/stac-extensions/raster/#raster-band-object

### Deprecated
- Specifying `class_name_mapping` by array is deprecated.
  Direct mapping as an object of index to class name should be used.
  For backward compatibility, mapping as array and using nested objects with `index` and `class_name` properties
  is still permitted, although overly verbose compared to the direct mapping.

### Removed
- Field `nodata_value`.
- Field `dtype`.

### Fixed
- Fixed references to other STAC extensions to use the official schema links on `https://stac-extensions.github.io/`.
- Fixed examples to refer to local files.
- Fixed formatting of tables and descriptions in README.

## [v1.0.0-beta2](https://github.com/crim-ca/dlm-extension/tree/v1.0.0-beta2)

### Added
- Initial release of the extension description and schema.

### Changed
- n/a

### Deprecated
- n/a

### Removed
- n/a

### Fixed
- n/a
