# ML Model Extension Specification

[![hackmd-github-sync-badge](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q/badge)](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q)

- **Title:** Model Extension
- **Identifier:** [https://schemas.stacspec.org/v1.0.0-beta.3/extensions/ml-model/json-schema/schema.json](https://schemas.stacspec.org/v1.0.0-beta.3/extensions/ml-model/json-schema/schema.json)
- **Field Name Prefix:** mlm
- **Scope:** Item, Collection
- **Extension Maturity Classification:** Proposal
- **Owner:**
  - [@sfoucher](https://github.com/sfoucher)
  - [@fmigneault](https://github.com/fmigneault)
  - [@ymoisan](https://github.com/ymoisan)
  - [@rbavery](https://github.com/rbavery)

This document explains the Template Extension to the [SpatioTemporal Asset Catalog (STAC)](https://github.com/radiantearth/stac-spec) specification. This document explains the fields of the STAC Model Extension to a STAC Item. The main objective of the extension is two-fold: 1) to enable building model collections that can be searched alongside associated STAC datasets and 2) to record all necessary parameters, artifact locations, and high level processing steps to deploy an inference service. Specifically, this extension records the following info to make ML models searchable and reusable:
1. Sensor band specifications
2. The two fundamental transforms on model inputs: rescale and normalization
3. Model output shape, data type, and its semantic interpretation
4. An optional, flexible description of the runtime environment to be able to run the model
5. Scientific references

Check the original technical report for an earlier version of the Model Extension [here](https://github.com/crim-ca/CCCOT03/raw/main/CCCOT03_Rapport%20Final_FINAL_EN.pdf) for more details.

![Image Description](https://i.imgur.com/cVAg5sA.png)

- Examples:
  - [Example with a ??? trained with torchgeo](examples/item.json) TODO update example
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json)
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name       | Type                                        | Description                                                            |
|------------------|---------------------------------------------|------------------------------------------------------------------------|
| dlm:input        | [ModelInput](#model-input)               | Describes the transformation between the EO data and the model input.  |
| dlm:architecture | [Architecture](#architecture) | Describes the model architecture.                                      |
| dlm:runtime      | [Runtime](#runtime)           | Describes the runtime environments to run the model (inference).       |
| dlm:output       | [ModelOutput](#model-output)             | Describes each model output and how to interpret it.                   |

In addition, fields from the following extensions must be imported in the item:
- [Scientific Extension Specification][stac-ext-sci] to describe relevant publications.
- [Version Extension Specification][stac-ext-ver] to define version tags.

[stac-ext-sci]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/scientific/README.md
[stac-ext-ver]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/version/README.md


### Model Input

| Field Name              | Type                            | Description                                                                                                                     |
|-------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                          | Informative name of the input variable. Example "RGB Time Series"                             |
| bands            | [Band](#bands)                       | Describes the EO data used to train or fine-tune the model.            |
| input_arrays           | [Array](#array) | Shape of the input arrays/tensors ($N \times C \times H \times W$).                                                                     |
| params           | dict | dictionary with names for the parameters and their values. some models may take scalars or other non-tensor inputs.                                                                     |
| scaling_factor          | number                          | Scaling factor to apply to get data within a `[0,1]` range. For instance `scaling_factor=0.004` for 8-bit data.                         |
| norm_by_channel          | string                          | Whether to normalize each channel by channel-wise statistics or to normalize by dataset statistics. "True" or "False" |
| norm_type          | string                          | Normalization method. Select one option from "min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "none" |
| rescale_type          | string                          | High level descriptor of the rescaling method to change image shape. Select one option from "crop", "pad", "interpolation", "none". If your rescaling method combines more than one of these operations, provide the name of the operation instead|
| statistics          | [Statistics](https://github.com/radiantearth/stac-spec/pull/1254/files#diff-2477b726f8c5d5d1c8b391be056db325e6918e78a24b414ccd757c7fbd574079R294)                          | Dataset statistics for the training dataset used to normalize the inputs. |
| pre_processing_function          | string                          | A url to the preprocessing function where normalization and rescaling takes place, and any other significant operations. Or, instead, the function code path, for example: my_python_module_name:my_processing_function|

#### Bands and Statistics

We use the [STAC 1.1 Bands Object](https://github.com/radiantearth/stac-spec/pull/1254) for representing bands information, including nodata value, data type, and common band names. Only bands used to train or fine tune the model should be included in this list.

A deviation is that we do not include the [Statistics](https://github.com/radiantearth/stac-spec/pull/1254/files#diff-2477b726f8c5d5d1c8b391be056db325e6918e78a24b414ccd757c7fbd574079R294) object at the band object level, but at the Model Input level. This is because in machine learning, we typically only need statistics for the dataset used to train the model in order to normalize any given bands input.

#### Array
| Field Name | Type   | Description                         |
|------------|--------|-------------------------------------|
| batch      | number | Batch size dimension (must be > 0). |
| time       | number | Number of timesteps  (must be > 0). |
| channels   | number | Number of channels  (must be > 0).  |
| height     | number | Height of the tensor (must be > 0). |
| width      | number | Width of the tensor (must be > 0).  |
|dim_order   | string | How the above dimensions are ordered with the tensor. "bhw", "bchw", "bthw", "btchw" |


### Architecture

| Field Name              | Type    | Description                                                 |
|-------------------------|---------|-------------------------------------------------------------|
| total_parameters        | integer | Total number of parameters.                                 |
| on_disk_size_mb         | number  | The memory size on disk of the model artifact (MB).         |
| ram_size_mb | number    | number  | The memory size in accelerator memory during inference (MB).|
| model_type                    | string  | Type of network (ex: ResNet-18).                            |
| summary                 | string  | Summary of the layers, can be the output of `print(model)`. |
| pretrained              | string  | Indicates the source of the pretraining (ex: ImageNet).     |

### Runtime

| Field Name            | Type                               | Description                                                                              |
|-----------------------|------------------------------------|------------------------------------------------------------------------------------------|
| framework             | string                             | Used framework (ex: PyTorch, TensorFlow).                                                |
| version               | string                             | Framework version (some models require a specific version of the framework).             |
| model_asset        | [Asset Object](https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object)                        | Common Metadata Collection level asset object containing URI to the model file.   |
| model_handler         | string                             | Inference execution function.                                                            |
| model_src_url         | string                             | Url of the source code (ex: GitHub repo).                                                |
| model_commit_hash     | string                             | Hash value pointing to a specific version of the code.                                   |
| docker                | [Container](#container) | Information for the deployment of the model in a docker instance.                        |
| batch_size_suggestion | number                             | A suggested batch size for a given compute instance type                                 |
| hardware_suggestion   | str                                | A suggested cloud instance type or accelerator model |

#### Container

| Field Name  | Type    | Description                                           |
|-------------|---------|-------------------------------------------------------|
| docker_file | string  | Url of the Dockerfile.                                |
| image_name  | string  | Name of the docker image.                             |
| tag         | string  | Tag of the image.                                     |
| working_dir | string  | Working directory in the instance that can be mapped. |
| run         | string  | Running command.                                      |
| accelerator         | boolean | True if the docker image requires a custom accelerator (CPU,TPU,MPS).              |

### Output

| Field Name               | Type                    | Description                                                                                                                                                                        |
|--------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task                     | [Task Enum](#task-enum) | Specifies the Machine Learning task for which the output can be used for.                                                                                                          |
| number_of_classes        | integer                 | Number of classes.                                                                                                                                                                 |
| final_layer_size         | \[integer]              | Sizes of the output tensor as ($N \times C \times H \times W$).                                                                                                                    |
| class_name_mapping       | dict                    | Mapping of the output index to a short class name, for each record we specify the index and the class name.                                                                        |
| post_processing_function          | string                          | A url to the postprocessing function where normalization and rescaling takes place, and any other significant operations. Or, instead, the function code path, for example: my_python_module_name:my_processing_function|


#### Task Enum

It is recommended to define `task` with one of the following values:
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

If the task falls within supervised machine learning and uses labels during training, this should align with the `label:tasks` values defined in [STAC Label Extension][stac-ext-label-props] for relevant
STAC Collections and Items employed with the model described by this extension.

[stac-ext-label-props]: https://github.com/stac-extensions/label#item-properties

## Relation types

The following types should be used as applicable `rel` types in the
[Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object).

| Type           | Description                           |
|----------------|---------------------------------------|
| fancy-rel-type | This link points to a fancy resource. |

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
