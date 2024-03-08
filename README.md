# Machine Learning Model Extension Specification

[![hackmd-github-sync-badge](https://hackmd.io/lekSD_RVRiquNHRloXRzeg/badge)](https://hackmd.io/lekSD_RVRiquNHRloXRzeg?both)

- **Title:** Machine Learning Model Extension
- **Identifier:** [https://schemas.stacspec.org/2.0.0.alpha.0/extensions/ml-model/json-schema/schema.json](https://schemas.stacspec.org/2.0.0.alpha.0/extensions/ml-model/json-schema/schema.json)
- **Field Name Prefix:** mlm
- **Scope:** Item, Collection
- **Extension Maturity Classification:** Proposal
- **Owner:**
  - [@fmigneault](https://github.com/fmigneault)
  - [@rbavery](https://github.com/rbavery)
  - [@ymoisan](https://github.com/ymoisan)
  - [@sfoucher](https://github.com/sfoucher)

The STAC Machine Learning Model (MLM) Extension provides a standard set of fields to describe machine learning models trained on overhead imagery and enable running model inference.

The main objectives of the extension are:

1) to enable building model collections that can be searched alongside associated STAC datasets
2) record all necessary bands, parameters, modeling artifact locations, and high-level processing steps to deploy an inference service.

Specifically, this extension records the following information to make ML models searchable and reusable:
1. Sensor band specifications
1. Model input transforms including resize and normalization
1. Model output shape, data type, and its semantic interpretation
1. An optional, flexible description of the runtime environment to be able to run the model
1. Scientific references

The MLM specification is biased towards providing metadata fields for supervised machine learning models. However, fields that relate to supervised ML are optional and users can use the fields they need for different tasks.

See [Best Practices](./best-practices.md) for guidance on what other STAC extensions you should use in conjunction with this extension. The Machine Learning Model Extension purposely omits and delegates some definitions to other STAC extensions to favor reusability and avoid metadata duplication whenever possible. A properly defined MLM STAC Item/Collection should almost never have the Machine Learning Model Extension exclusively in `stac_extensions`.

Check the original technical report for an earlier version of the Model Extension, formerly known as the Deep Learning Model Extension (DLM), [here](https://github.com/crim-ca/CCCOT03/raw/main/CCCOT03_Rapport%20Final_FINAL_EN.pdf) for more details. The DLM was renamed to the current MLM Extension and refactored to form a cohesive definition across all machine learning approaches, regardless of whether the approach constitutes a deep neural network or other statistical approach.

![Image Description](https://i.imgur.com/cVAg5sA.png)

- Examples:
  - [Example with a ??? trained with torchgeo](examples/item.json) TODO update example
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json) TODO update
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name            | Type                                          | Description                                                                                                                                                                                                                                                                                                                                          |
|-----------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlm:name              | string                                        | **REQUIRED.** A unique name for the model. This should include but be distinct from simply naming the model architecture. If there is a publication or other published work related to the model, use the official name of the model.                                                                                                                |
| mlm:task              | [Task Enum](#task-enum)                       | **REQUIRED.** Specifies the primary Machine Learning task for which the output can be used for. If there are multi-modal outputs, specify the primary task and specify each task in the [Model Output Object](#model-output-object).                                                                                                                 |
| mlm:framework         | string                                        | **REQUIRED.** Framework used to train the model (ex: PyTorch, TensorFlow).                                                                                                                                                                                                                                                                           |
| mlm:framework_version | string                                        | **REQUIRED.** The `framework` library version. Some models require a specific version of the machine learning `framework` to run.                                                                                                                                                                                                                    |
| mlm:file_size         | integer                                       | **REQUIRED.** The size on disk of the model artifact (bytes).                                                                                                                                                                                                                                                                                        |
| mlm:memory_size       | integer                                       | **REQUIRED.** The in-memory size of the model on the accelerator during inference (bytes).                                                                                                                                                                                                                                                           |
| mlm:input             | [[Model Input Object](#model-input-object)]   | **REQUIRED.** Describes the transformation between the EO data and the model input.                                                                                                                                                                                                                                                                  |
| mlm:output            | [[Model Output Object](#model-output-object)] | **REQUIRED.** Describes each model output and how to interpret it.                                                                                                                                                                                                                                                                                   |
| mlm:runtime           | [[Runtime Object](#runtime-object)]           | **REQUIRED.** Describes the runtime environment(s) to run inference with the model asset(s).                                                                                                                                                                                                                                                         |
| mlm:total_parameters  | integer                                       | Total number of model parameters, including trainable and non-trainable parameters.                                                                                                                                                                                                                                                                  |
| mlm:pretrained_source | string                                        | The source of the pretraining. Can refer to popular pretraining datasets by name (i.e. Imagenet) or less known datasets by URL and description.                                                                                                                                                                                                      |
| mlm:summary           | string                                        | Text summary of the model and it's purpose.                                                                                                                                                                                                                                                                                                          |
| mlm:parameters        | [Parameters Object](#params-object)           | Mapping with names for the parameters and their values. Some models may take additional scalars, tuples, and other non-tensor inputs like text during inference (Segment Anything). The field should be specified here if parameters apply to all Model Input Objects. If each Model Input Object has parameters, specify parameters in that object. |

In addition, fields from the following extensions must be imported in the item:
- [Scientific Extension Specification][stac-ext-sci] to describe relevant publications.
- [Version Extension Specification][stac-ext-ver] to define version tags.

[stac-ext-sci]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/scientific/README.md
[stac-ext-ver]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/version/README.md

### Model Input Object


| Field Name              | Type                                                                             | Description                                                                                                                                                                                                                                        |   |
|-------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| name                    | string                                                                           | **REQUIRED.** Informative name of the input variable. Example "RGB Time Series"                                                                                                                                                                    |   |
| bands                   | [string]                                                                         | **REQUIRED.** The names of the raster bands used to train or fine-tune the model, which may be all or a subset of bands available in a STAC Item's [Band Object](#bands-and-statistics).                                                           |   |
| input_array             | [Array Object](#feature-array-object)                                            | **REQUIRED.** The N-dimensional array object that describes the shape, dimension ordering, and data type.                                                                                                                                          |   |
| parameters              | [Parameters Object](#params-object)                                              | Mapping with names for the parameters and their values. Some models may take additional scalars, tuples, and other non-tensor inputs like text.                                                                                                    |   |
| norm_by_channel         | boolean                                                                          | Whether to normalize each channel by channel-wise statistics or to normalize by dataset statistics. If True, use an array of [Statistics Objects](#bands-and-statistics) that is ordered like the `bands` field in this object.                    |   |
| norm_type               | string                                                                           | Normalization method. Select one option from "min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "norm_with_clip", "none"                                                                                                                                |   |
| resize_type            | string                                                                           | High-level descriptor of the rescaling method to change image shape. Select one option from "crop", "pad", "interpolation", "none". If your rescaling method combines more than one of these operations, provide the name of the operation instead |   |
| statistics              | [Statistics Object](stac-statistics) `\|` [[Statistics Object](stac-statistics)] | Dataset statistics for the training dataset used to normalize the inputs.                                                                                                                                                                          |   |
| norm_with_clip_values              | [integer] |  If norm_type = "norm_with_clip" this array supplies a value that is less than the band maximum. The array must be the same length as "bands", each value is used to divide each band before clipping values between 0 and 1.                                                                                                                                                                        |
| pre_processing_function | string                                                                           | A url to the preprocessing function where normalization and rescaling takes place, and any other significant operations. Or, instead, the function code path, for example: `my_python_module_name:my_processing_function`                          |   |

#### Parameters Object

| Field Name                            | Type                         | Description                                                                                                                                                                                                                                                                      |
|---------------------------------------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *parameter names depend on the model* | number `\|` string `\|` boolean `\|` array | The number of fields and their names depend on the model. Values should not be n-dimensional array inputs. If the model input can be represented as an n-dimensional array, it should instead be supplied as another [model input object](#model-input-object). |

The parameters field can either be specified in the [Model Input Object](#model-input-object) if they are associated with a specific input or as an [Item or Collection](#item-properties-and-collection-fields) field if the parameters are supplied without relation to a specific model input.

#### Bands and Statistics

We use the [STAC 1.1 Bands Object](https://github.com/radiantearth/stac-spec/pull/1254) for representing bands information, including the nodata value, data type, and common band names. Only bands used to train or fine tune the model should be included in this `bands` field.

A deviation from the [STAC 1.1 Bands Object](https://github.com/radiantearth/stac-spec/pull/1254) is that we do not include the [Statistics](stac-statistics) object at the band object level, but at the Model Input level. This is because in machine learning, it is common to only need overall statistics for the dataset used to train the model to normalize all bands.

[stac-statistics]: https://github.com/radiantearth/stac-spec/pull/1254/files#diff-2477b726f8c5d5d1c8b391be056db325e6918e78a24b414ccd757c7fbd574079R294

#### Array Object

| Field Name | Type      | Description                                                                                                                                                                                                                                                                                                                                                                                            |  |
|------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| shape      | [integer] | **REQUIRED.** Shape of the input n-dimensional array ($N \times C \times H \times W$), including the batch size dimension. The batch size dimension must either be greater than 0 or -1 to indicate an unspecified batch dimension size.                                                                                                                                                               |  |
| dim_order  | string    | **REQUIRED.** How the above dimensions are ordered within the `shape`. "bhw", "bchw", "bthw", "btchw" are valid orderings where b=batch, c=channel, t=time, h=height, w=width.                                                                                                                                                                                                                         |  |
| data_type  | enum      | **REQUIRED.** The data type of values in the n-dimensional array. For model inputs, this should be the data type of the processed input supplied to the model inference function, not the data type of the source bands. Use one of the [common metadata data types](https://github.com/stac-extensions/raster?tab=readme-ov-file#data-types). |  |

Note: It is common in the machine learning, computer vision, and remote sensing communities to refer to rasters that are inputs to a model as arrays or tensors. Array Objects are distinct from the JSON array type used to represent lists of values.


### Runtime Object

| Field Name              | Type                                  | Description                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_asset             | [Asset Object](stac-asset)            | **REQUIRED.** Asset object containing URI to the model file. Recommended asset `roles` include `weights` for model weights that need to be loaded by a model definition and `compiled` for models that can be loaded directly without an intermediate model definition.                                                                                                      |
| source_code             | [Asset Object](stac-asset)            | **REQUIRED.** Source code description. Can describe a github repo, zip archive, etc. The `description` field in the Asset Object should reference the inference function, for example my_package.my_module.predict. Recommended asset `roles` include `code` and `metadata`, since the source code asset might also refer to more detailed metadata than this spec captures. |
| accelerator             | [Accelerator Enum](#accelerator-enum) | **REQUIRED.** The intended computational hardware that runs inference.                                                                                                                                                                                                                                                                                                       |
| accelerator_constrained | boolean                               | **REQUIRED.** True if the intended `accelerator` is the only `accelerator` that can run inference. False if other accelerators, such as amd64 (CPU), can run inference.                                                                                                                                                                                                      |
| hardware_summary        | string                                | **REQUIRED.** A high level description of the number of accelerators, specific generation of the `accelerator`, or other relevant inference details.                                                                                                                                                                                                                         |
| container               | [Container](#container)               | **RECOMMENDED.** Information to run the model in a container instance.                                                                                                                                                                                                                                                                                                       |
| commit_hash             | string                                | Hash value pointing to a specific version of the code used to run model inference. If this is supplied, `source code` should also be supplied and the commit hash must refer to a Git repository linked or described in the `source_code` [Asset Object](stac-asset).                                                                                                        |
| batch_size_suggestion   | number                                | A suggested batch size for the accelerator and summarized hardware.                                                                                                                                                                                                                                                                                                          |

#### Accelerator Enum

It is recommended to define `accelerator` with one of the following values:

- `amd64` models compatible with AMD or Intel CPUs (no hardware specific optimizations)
- `cuda` models compatible with NVIDIA GPUs
- `xla` models compiled with XLA. models trained on TPUs are typically compiled with XLA.
- `amd-rocm` models trained on AMD GPUs
- `intel-ipex-cpu` for models optimized with IPEX for Intel CPUs
- `intel-ipex-gpu` for models optimized with IPEX for Intel GPUs
- `macos-arm` for models trained on Apple Silicon

[stac-asset]: https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object

#### Container Object

| Field Name     | Type   | Description                                           |
|----------------|--------|-------------------------------------------------------|
| container_file | string | Url of the container file (Dockerfile).               |
| image_name     | string | Name of the container image.                          |
| tag            | string | Tag of the image.                                     |
| working_dir    | string | Working directory in the instance that can be mapped. |
| run            | string | Running command.                                      |

If you're unsure how to containerize your model, we suggest starting from the latest official container image for your framework that works with your model and pinning the container version.

Examples:
[Pytorch Dockerhub](https://hub.docker.com/r/pytorch/pytorch/tags)
[Pytorch Docker Run Example](https://github.com/pytorch/pytorch?tab=readme-ov-file#docker-image)

[Tensorflow Dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/tags?page=8&ordering=last_updated)
[Tensorflow Docker Run Example](https://www.tensorflow.org/install/docker#gpu_support)

Using a base image for a framework looks like


```dockerfile
# In your Dockerfile, pull the latest base image with all framework dependencies including accelerator drivers
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

### Your specific environment setup to run your model
RUN pip install my_package
```

You can also use other base images. Pytorch and Tensorflow offer docker images for serving models for inference.
- [Torchserve](https://pytorch.org/serve/)
- [TFServing](https://github.com/tensorflow/serving)

### Model Output Object

| Field Name               | Type                                          | Description                                                                                                                                                                                            |
|--------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task                     | [Task Enum](#task-enum)                       | **REQUIRED.** Specifies the Machine Learning task for which the output can be used for.                                                                                                                |
| result_array                   | [[Result Array Object](#result-array-object)] | The list of output arrays/tensors from the model.                                             |
| classification:classes       | [[Class Object](#class-object)]       | A list of class objects adhering to the [Classification extension](https://github.com/stac-extensions/classification).                                                                                                                      |
| post_processing_function | string                                        | A url to the postprocessing function where normalization, rescaling, and other operations take place.. Or, instead, the function code path, for example: `my_package.my_module.my_processing_function` |

While only `task` is a required field, all fields are recommended for supervised tasks that produce a fixed shape tensor and have output classes.
`image-captioning`, `multi-modal`, and `generative` tasks may not return fixed shape tensors or classes.

#### Task Enum

It is recommended to define `task` with one of the following values for each Model Output Object:
- `regression`
- `classification`
- `object detection`
- `semantic segmentation`
- `instance segmentation`
- `panoptic segmentation`
- `multi-modal`
- `similarity search`
- `image captioning`
- `generative`
- `super resolution`

If the task falls within the category of supervised machine learning and uses labels during training, this should align with the `label:tasks` values defined in [STAC Label Extension][stac-ext-label-props] for relevant
STAC Collections and Items published with the model described by this extension.

[stac-ext-label-props]: https://github.com/stac-extensions/label#item-properties

#### Result Array Object

| Field Name | Type      | Description                                                                                                                                                                                                                                                                                                                                                          |
|------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | [integer] | **REQUIRED.** Shape of the n-dimensional result array ($N \times H \times W$), possibly including a batch size dimension. The batch size dimension must either be greater than 0 or -1 to indicate an unspecified batch dimension size.                                                                                                                              |
| dim_names  | [string]  | **REQUIRED.** The names of the above dimensions of the result array, ordered the same as this object's `shape` field.                                                                                                                                                                                                                                                |
| data_type  | enum      | **REQUIRED.** The data type of values in the n-dimensional array. For model outputs, this should be the data type of the result of the model inference  without extra post processing. Use one of the [common metadata data types](https://github.com/radiantearth/stac-spec/blob/f9b3c59ba810541c9da70c5f8d39635f8cba7bcd/item-spec/common-metadata.md#data-types). |



#### Class Object

See the documentation for the [Class Object](https://github.com/stac-extensions/classification?tab=readme-ov-file#class-object). We don't use the Bit Field Object since inputs and outputs to machine learning models don't typically use bit fields.

## Relation types

The following types should be used as applicable `rel` types in the
[Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object) of STAC Items describing Band Assets used with a model.

| Type         | Description                                                                                                                |
|--------------|----------------------------------------------------------------------------------------------------------------------------|
| derived_from | This link points to <model>_item.json or <model>_collection.json.  Replace <model> with the unique [`mlm:name`](#item-properties-and-collection-fields) field's value. |

## Contributing

All contributions are subject to the
[STAC Specification Code of Conduct][stac-spec-code-conduct].
For contributions, please follow the
[STAC specification contributing guide][stac-spec-contrib-guide] Instructions
for running tests are copied here for convenience.

[stac-spec-code-conduct]: https://github.com/radiantearth/stac-spec/blob/master/CODE_OF_CONDUCT.md
[stac-spec-contrib-guide]: https://github.com/radiantearth/stac-spec/blob/master/CONTRIBUTING.md

### Running tests

The same checks that run as checks on PRs are part of the repository and can be run locally to verify that changes are valid. To run tests locally, you'll need `npm`, which is a standard part of any [node.js](https://nodejs.org/en/download/) installation.

First, install everything with npm once. Navigate to the root of this repository and on your command line run:

```bash
npm install
```

Then to check Markdown formatting and test the examples against the JSON schema, you can run:
```bash
npm test
```

This will spit out the same texts that you see online, and you can then go and fix your markdown or examples.

If the tests reveal formatting problems with the examples, you can fix them with:
```bash
npm run format-examples
```
