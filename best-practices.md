# ML Model Extension Best Practices

This document makes a number of recommendations for creating real world ML Model Extensions. None of them are required to meet the core specification, but following these practices will improve the documentation of your model and make life easier for client tooling and users. They come about from practical experience of implementors and introduce a bit more 'constraint' for those who are creating STAC objects representing their models or creating tools to work with STAC.

## Recommended Extensions to Compose with the ML Model Extension

### Processing Extension

We recommend using the `processing:lineage` and `processing:level` fields from the [Processing Extension](https://github.com/stac-extensions/processing) to make it clear how [Model Input Objects](./README.md#model-input-object) are processed.

For example:

TODO supply example

TODO provide other suggestions on extensions to compose with this one. STAC ML AOI, STAC Label, ...
