# Template Extension Specification

- **Title:** Template
- **Identifier:** <https://stac-extensions.github.io/template/v1.0.0/schema.json>
- **Field Name Prefix:** template
- **Scope:** Item, Collection
- **Extension [Maturity Classification](https://github.com/radiantearth/stac-spec/tree/master/extensions/README.md#extension-maturity):** Proposal
- **Owner**: @your-gh-handles @person2

This document explains the Template Extension to the [SpatioTemporal Asset Catalog](https://github.com/radiantearth/stac-spec) (STAC) specification.
This is the place to add a short introduction.

- Examples:
  - [Item example](examples/item.json): Shows the basic usage of the extension in a STAC Item
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json)
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name           | Type                      | Description |
| -------------------- | ------------------------- | ----------- |
| template:new_field   | string                    | **REQUIRED**. Describe the required field... |
| template:xyz         | [XYZ Object](#xyz-object) | Describe the field... |
| template:another_one | \[number]                 | Describe the field... |

### Additional Field Information

#### template:new_field

This is a much more detailed description of the field `template:new_field`...

### XYZ Object

This is the introduction for the purpose and the content of the XYZ Object...

| Field Name  | Type   | Description |
| ----------- | ------ | ----------- |
| x           | number | **REQUIRED**. Describe the required field... |
| y           | number | **REQUIRED**. Describe the required field... |
| z           | number | **REQUIRED**. Describe the required field... |

## Relation types

The following types should be used as applicable `rel` types in the
[Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object).

| Type                | Description |
| ------------------- | ----------- |
| fancy-rel-type      | This link points to a fancy resource. |

## Contributing

All contributions are subject to the
[STAC Specification Code of Conduct](https://github.com/radiantearth/stac-spec/blob/master/CODE_OF_CONDUCT.md).
For contributions, please follow the
[STAC specification contributing guide](https://github.com/radiantearth/stac-spec/blob/master/CONTRIBUTING.md) Instructions
for running tests are copied here for convenience.

### Running tests

The same checks that run as checks on PR's are part of the repository and can be run locally to verify that changes are valid. 
To run tests locally, you'll need `npm`, which is a standard part of any [node.js installation](https://nodejs.org/en/download/).

First you'll need to install everything with npm once. Just navigate to the root of this repository and on 
your command line run:
```bash
npm install
```

Then to check markdown formatting and test the examples against the JSON schema, you can run:
```bash
npm test
```

This will spit out the same texts that you see online, and you can then go and fix your markdown or examples.

If the tests reveal formatting problems with the examples, you can fix them with:
```bash
npm run format-examples
```
