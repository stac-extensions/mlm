# ML Model Extension Best Practices

This document makes a number of recommendations for creating real world ML Model Extensions.
None of them are required to meet the core specification, but following these practices will improve the documentation
of your model and make life easier for client tooling and users. They come about from practical experience of
implementors and introduce a bit more 'constraint' for those who are creating STAC objects representing their 
models or creating tools to work with STAC.

- [Using STAC Common Metadata Fields for the ML Model Extension](#using-stac-common-metadata-fields-for-the-ml-model-extension)
- [Recommended Extensions to Compose with the ML Model Extension](#recommended-extensions-to-compose-with-the-ml-model-extension)
  - [Processing Extension](#processing-extension)
  - [ML-AOI and Label Extensions](#ml-aoi-and-label-extensions)
  - [Classification Extension](#classification-extension)
  - [Scientific Extension](#scientific-extension)
  - [File Extension](#file-extension)
  - [Example Extension](#example-extension)
  - [Version Extension](#version-extension)

## Using STAC Common Metadata Fields for the ML Model Extension

It is recommended to use the `start_datetime` and `end_datetime`, `geometry`, and `bbox` in a STAC Item,
and the corresponding
[Extent Object](https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#extent-object)
in a Collection, to represent the *recommended context* of the data the model was trained with and for which the model
should have appropriate domain knowledge for inference.

For example, if a model was trained using the [EuroSAT][EuroSAT-github] dataset, and represented using MLM, it would
be reasonable to describe it with a time range of 2015-2018 and an area corresponding to the European Urban Atlas, as
described by the [EuroSAT paper][EuroSAT-paper]. However, it could also be considered adequate to define a wider extent,
considering that it would not be unexpected to have reasonably similar classes and domain distribution in following
years and in other locations. Provided that the exact extent applicable for a model is difficult to define reliably,
it is left to the good judgement of users to provide adequate values. Note that users employing the model can also
choose to apply it for contexts outside the *recommended* extent for the same reason.

[EuroSAT-github]: https://github.com/phelber/EuroSAT
[EuroSAT-paper]: https://www.researchgate.net/publication/319463676

As another example, let us consider a model which is trained on imagery from all over the world
and is robust enough to be applied to any time period. In this case, the common metadata to use with the model
could include the bbox of "the world" `[-90, -180, 90, 180]` and the `start_datetime` and `end_datetime` range
would ideally be generic values like `["1900-01-01T00:00:00Z", null]` (see warning below).
However, due to limitations with the STAC 1.0 specification, this time extent is not applicable.

> [!WARNING]
> The `null` value is not allowed for datetime specification.
> As a workaround, the `end_datetime` can be set with a "very large value"
> (similarly to `start_datetime` set with a small value), such as `"9999-12-31T23:59:59Z"`.
> Alternatively, the model can instead be described with only `datetime` corresponding to its publication date.
> <br><br>
> For more details, see the following [discussion](https://github.com/radiantearth/stac-spec/issues/1268).

It is to be noted that generic and very broad spatiotemporal
extents like above rarely reflect the reality regarding the capabilities and precision of the model to predict reliable
results. If a more restrained area and time of interest can be identified, such as the ranges for which the training
dataset applies, or a test split dataset that validates the applicability of the model on other domains, those should
be provided instead. Nevertheless, users of the model are still free to apply it outside the specified extents.

If specific datasets with training/validation/test splits are known to support the claims of the suggested extent for
the model, it is recommended that they are included as reference to the STAC Item/Collection using MLM. For more
information regarding these references, see the [ML-AOI and Label Extensions](#ml-aoi-and-label-extensions) details.

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
    "mlm:name": "<name-in-MLM-STAC-Item",
    "processing:level": "L4",
    "processing:expression": {
      "format": "stac-mlm",
      "expression": "<URI-to-MLM-STAC-Item>"
    }
  }
}
```

Furthermore, the STAC Item representing the derived product could also include
a [Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object)
referring back to the MLM definition using `rel: derived_from`, as described in
[MLM Relation Types](README.md#relation-types). Such a link would like something like the following:

```json
{
  "links": [
    {
      "rel": "derived_from",
      "type": "application/geo+json",
      "href": "<URI-to-MLM-STAC-Item>",
      "mlm:name": "<name-in-MLM-STAC-Item",
      "processing:level": "L4"
    }
  ]
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
leading to a new MLM STAC Item definition (see also [STAC Version Extension](#version-extension)).

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

### Classification Extension

Since it is expected that a model will provide some kind of classification values as output, the 
[Classification Extension](https://github.com/stac-extensions/classification) can be leveraged inside
MLM definition to indicate which class values can be contained in the resulting output from the model prediction.

For more details, see the [Model Output Object](README.md#model-output-object) definition.

> [!NOTE]
> Update according to https://github.com/stac-extensions/classification/issues/48

### Scientific Extension

Provided that most models derive from previous scientific work, it is strongly recommended to employ the 
[Scientific Extension](https://github.com/stac-extensions/scientific) to provide references corresponding to the
original source of the model (`sci:doi`, `sci:citation`). This can help users find more information about the model,
its underlying architecture, or ways to improve it by piecing together the related work (`sci:publications`) that
lead to its creation.

This extension can also be used for the purpose of publishing new models, by providing to users the necessary details
regarding how they should cite its use (i.e.: `sci:citation` field and `cite-as` relation type).

### File Extension

In order to provide a reliable and reproducible machine learning pipeline, external references to data required by the
model should employ the [file](https://github.com/stac-extensions/file?tab=readme-ov-file#asset--link-object-fields) to
validate that they are properly retrieved for inference.

One of the most typical case is the definition of an external file reference to model weights, often stored on a
Git LFS or S3 bucket due to their size. Providing the `file:checksum` and `file:size` for this file can help ensure
that the model is properly instantiated from the expected weights, or that sufficient storage is allocated to run it.

```json
{
  "stac_extensions": [
    "https://stac-extensions.github.io/mlm/v1.0.0/schema.json",
    "https://stac-extensions.github.io/file/v2.1.0/schema.json"
  ],
  "assets": {
    "model": {
      "type": "application/x-pytorch",
      "href": "<URI-to-model-weights>",
      "roles": [
        "mlm:model",
        "mlm:weights",
        "data"
      ],
      "file:size": 123456789,
      "file:checksum": "12209f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      "mlm:artifact_type": "torch.save"
    }
  }
}
```

### Example Extension

In order to help users understand how to apply and run the described machine learning model,
the [Example Extension](https://github.com/stac-extensions/example-links#fields) can be used to provide code examples
demonstrating how it can be applied.

For example, a [Model Card on Hugging Face](https://huggingface.co/docs/hub/en/model-cards)
is often provided (see [Hugging Face Model examples](https://huggingface.co/models)) to describe the model, which
can embed sample code and references to more details about the model. This kind of reference should be added under
the `links` of the STAC Item using MLM.

Typically, a STAC Item using the MLM extension to describe the training or
inference strategies to apply a model should define the [Source Code Asset](README.md#source-code-asset).
This code is in itself ideal to guide users how to run it, and should therefore be replicated as an `example` link
reference to offer more code samples to execute the model.

> [!NOTE]
> Update according to https://github.com/stac-extensions/example-links/issues/4

### Version Extension

In the even that a model is retrained with gradually added annotations or improved training strategies leading to
better performances, the existing model and newer models represented by STAC Items with MLM should also make use of
the [Version Extension](https://github.com/stac-extensions/version). Using the fields and link relation types defined
by this extension, the retraining cycle of the model can better be described, with a full history of the newer versions
developed.

Additionally, the `version:experimental` field should be considered for models being trained and still under evaluation
before widespread deployment. This can be particularly useful for annotating models experiments during cross-validation
training process to find the "best model". This field could also be used to indicate if a model is provided for
educational purposes only.
