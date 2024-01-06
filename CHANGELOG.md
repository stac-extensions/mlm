# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- more Task Enum tasks
- accelerator options
- batch_size and hardware suggestion
- ram_size_mb to specify model ram requirements during inference
- added time to the Tensor object as an optional dim
- Use common metadata Asset Object to refer to Model Artifact and artifact metadata as a Collection level object

### Changed
- selected_bands > band_names, the same human readable names used in the common metadata band objects.
- replaced normalization:mean, etc. with statistics from STAC 1.1 common metadata
- added pydantic models for internal schema objects in stac_model package and published to PYPI

[raster-band-object]: https://github.com/stac-extensions/raster/#raster-band-object

### Deprecated
-

### Removed
- Data Object, replaced with common metadata band object which also records data_type and nodata type

# TODO link release here

## [Unreleased]

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
