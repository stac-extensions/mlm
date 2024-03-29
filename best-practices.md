# ML Model Extension Best Practices

This document makes a number of recommendations for creating real world ML Model Extensions.
None of them are required to meet the core specification, but following these practices will improve the documentation
of your model and make life easier for client tooling and users. They come about from practical experience of
implementors and introduce a bit more 'constraint' for those who are creating STAC objects representing their 
models or creating tools to work with STAC.

## Using STAC Common Metadata Fields for the ML Model Extension

It is recommended to use the `start_datetime` and `end_datetime`, `geometry`, and `bbox` to represent the 
recommended context of data the model was trained with and for which the model should have appropriate domain
knowledge for inference. For example, we can consider a model which is trained on imagery from all over the world
and is robust enough to be applied to any time period. In this case, the common metadata to use with the model
would include the bbox of "the world" `[-90, -180, 90, 180]` and the `start_datetime` and `end_datetime` range could
be generic values like `["1900-01-01", null]`.

## Recommended Extensions to Compose with the ML Model Extension

### Processing Extension

It is recommended to use at least the `processing:lineage` and `processing:level` fields from 
the [Processing Extension](https://github.com/stac-extensions/processing) to make it clear 
how [Model Input Objects](./README.md#model-input-object) are processed by the data provider prior to an
inference preprocessing pipeline. This can help users locate the correct version of the dataset used during model
inference or help them reproduce the data processing pipeline.

For example:

```json
{
  "processing:lineage": "GRD Post Processing",
  "processing:level": "L1C",
  "processing:facility": "Copernicus S1 Core Ground Segment - DPA",
  "processing:software": {
    "Sentinel-1 IPF": "002.71"
  }
}
```

STAC Items or STAC Assets resulting from the model inference should be
annotated with [`processing:level = L4`](https://github.com/stac-extensions/processing?tab=readme-ov-file#suggested-processing-levels)
(as described below) to indicate that they correspond from the output of an ML model.

> <b>processing:level = L4</b><br>
> Model output or results from analyses of lower level data (i.e.: variables that are not directly measured by the instruments, but are derived from these measurements)

Furthermore, the [`processing:expression`](https://github.com/stac-extensions/processing?tab=readme-ov-file#expression-object)
should be specified with a reference to the STAC Item employing the MLM extension to provide full context of the source
of the derived product.

A potential representation of a STAC Asset could be as follows: 
```json
{
  "model-output": {
    "processing:level": "L4",
    "processing:expression": {
      "format": "stac-mlm",
      "expression": "<URI-to-MLM-STAC-Item>"
    }
  }
}
```

### ML-AOI and Label Extensions

Supervised machine learning models will typically employ a dataset of training, validation and test samples.
If those samples happen to be represented by STAC Collections and Items annotated with
the [ML-AOI Extension](https://github.com/stac-extensions/ml-aoi), notably with the corresponding `ml-aoi:split`
and all their annotations with [Label Extension](https://github.com/stac-extensions/label) references, the STAC Item
that contains the MLM Extension should include those STAC Collections in its `links` listing in order
to provide direct references to the training dataset that was employed for creating the model.

Providing dataset references would, in combination with the training pipeline contained under an
[MLM Asset Object](README.md#assets-objects) annotated by the `mlm:training-runtime` role,
allow users to retrain the model for validation, or with adaptations to improve it, eventually
leading to a new MLM STAC Item definition.

```json
{
  "id": "stac-item-model",
  "stac_extensions": [
    "https://stac-extensions.github.io/mlm/v1.0.0/schema.json",
    "https://stac-extensions.github.io/ml-aoi/v0.2.0/schema.json"
  ],
  "assets": {
    "mlm:training": {
      "title": "Model Training Pipeline",
      "href": "docker.io/training/image:latest",
      "type": "application/vnd.oci.image.index.v1+json",
      "roles": ["mlm:training-runtime"]
    }
  },
  "links": [
    {
      "rel": "derived_from",
      "type": "application/json",
      "href": "<URI-to-STAC-Collection-Split-Train>",
      "ml-aoi:split": "train"
    },
    {
      "rel": "derived_from",
      "type": "application/json",
      "href": "<URI-to-STAC-Collection-Split-Valid>",
      "ml-aoi:split": "validate"
    },
    {
      "rel": "derived_from",
      "type": "application/json",
      "href": "<URI-to-STAC-Collection-Split-Test>",
      "ml-aoi:split": "test"
    }
  ]
}
```
