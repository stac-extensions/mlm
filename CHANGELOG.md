# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- more [Task Enum](./README.md#task-enum) tasks
- [accelerator](./README#accelerators) options in [Runtime Object](./README#runtime-object)
- [Model Output Object](./README.md#model-output-object)
- batch_size and hardware summary
- [`disk_size`, `memory_size`](./README#architecture-object)
- [`hardware_summary`, `accelerator`, `accelerator_constrained`](./README#runtime-object) to specify hardware requirements for inference
- Use common metadata [Asset Object](https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object) to refer to model asset and source code.
- use `classification:classes` in Model Output
- add `scene-classification` to the Enum Tasks to allow disambiguation between pixel-wise and patch-based classification

### Changed
- reorganized `dlm:architecture` nested fields to exist at the top level of properties as `mlm:name`, `mlm:summary`
  and so on to provide STAC API search capabilities.
- replaced `normalization:mean`, etc. with [statistics](./README.md#bands-and-statistics) from STAC 1.1 common metadata
- added `pydantic` models for internal schema objects in `stac_model` package and published to PYPI
- specified [rel_type](./README.md#relation-types) to be `derived_from` and specify how model item or collection json should be named
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
- Data Object, replaced with [Model Input Object](./README.md#model-input-object) that uses the `name` field from
  the [common metadata band object](https://github.com/radiantearth/stac-spec/blob/f9b3c59ba810541c9da70c5f8d39635f8cba7bcd/item-spec/common-metadata.md#bands)
  which also records `data_type` and `nodata` type

# TODO link release here

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

[v1.0.0-beta2]: <https://github.com/sfoucher/dlm-extension/compare/v1.0.0...HEAD>

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
