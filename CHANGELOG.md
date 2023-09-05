# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed
- Modified `$id` if the extension schema to refer to the expected location when eventually released
  (`https://schemas.stacspec.org/v1.0.0-beta.3/extensions/dl-model/json-schema/schema.json`).
- Replaced `dtype` field by `data_type` to better align with the corresponding field of
  [`raster:bands`][raster-band-object].
- Replaced `nodata_value` field by `nodata` to better align with the corresponding field of
  [`raster:bands`][raster-band-object].

[raster-band-object]: https://github.com/stac-extensions/raster/#raster-band-object

### Deprecated

### Removed

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
