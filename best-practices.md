# ML Model Extension Best Practices

This document makes a number of recommendations for creating real world ML Model Extensions. None of them are required to meet the core specification, but following these practices will improve the documentation of your model and make life easier for client tooling and users. They come about from practical experience of implementors and introduce a bit more 'constraint' for those who are creating STAC objects representing their models or creating tools to work with STAC.

## Using STAC Common Metadata Fields for the ML Model Extension

We recommend using the `start_datetime` and `end_datetime`, `geometry`, and `bbox` to represent the recommended context of data the model was trained with and for which the model should have appropriate domain knowledge for inference. For example, we can consider a model which is trained on imagery from all over the world and is robust enough to be applied to any time period. In this case, the common metadata to use with the model would include the bbox of "the world" `[-90, -180, 90, 180]` and the start_datetime and end_datetime range could be generic values like `["1900-01-01", null]`.

## Recommended Extensions to Compose with the ML Model Extension

### Processing Extension

We recommend using at least the `processing:lineage` and `processing:level` fields from the [Processing Extension](https://github.com/stac-extensions/processing) to make it clear how [Model Input Objects](./README.md#model-input-object) are processed by the data provider prior to an inference preprocessing pipeline. This can help users locate the correct version of the dataset used during model inference or help them reproduce the data processing pipeline.

For example:

```
"processing:lineage": "GRD Post Processing",
"processing:level": "L1C",
"processing:facility": "Copernicus S1 Core Ground Segment - DPA",
"processing:software": {
    "Sentinel-1 IPF": "002.71"
}
```

STAC Items or STAC Assets with asset properties resulting from the model inference should be annotated with [`processing:level = L4`](https://github.com/stac-extensions/processing?tab=readme-ov-file#suggested-processing-levels).

> Model output or results from analyses of lower level data (i.e.,variables that are not directly measured by the instruments, but are derived from these measurements)

TODO provide other suggestions on extensions to compose with this one. STAC ML AOI, STAC Label, ...
