# Deep Learning Model Extension Specification

![hackmd-github-sync-badge](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q/badge)](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q)

- **Title:** Deep Learning Model Extension
- **Identifier:** [https://schemas.stacspec.org/v1.0.0-beta.3/extensions/dl-model/json-schema/schema.json](https://schemas.stacspec.org/v1.0.0-beta.3/extensions/dl-model/json-schema/schema.json)
- **Field Name Prefix:** dlm
- **Scope:** Item, Collection
- **Extension Maturity Classification:** Proposal
- **Owner:**
  - [@sfoucher](https://github.com/sfoucher)
  - [@fmigneault](https://github.com/fmigneault)
  - [@ymoisan](https://github.com/ymoisan)

This document explains the Template Extension to the [SpatioTemporal Asset Catalog (STAC)](https://github.com/radiantearth/stac-spec) specification. This document explains the fields of the STAC Deep Learning Model (dlm) Extension to a STAC Item. The main objective is to be able to build model collections that can be searched and that contain enough information to be able to deploy an inference service. When Deep Learning models are trained using satellite imagery, it is important to track essential information if you want to make them searchable and reusable:
1. Input data origin and specifications
2. Model basic transforms: rescale and normalization
3. Model output and its semantic interpretation
4. Runtime environment to be able to run the model
5. Scientific references

Check the original technical report [here](https://github.com/crim-ca/CCCOT03/raw/main/CCCOT03_Rapport%20Final_FINAL_EN.pdf) for more details.

![Image Description](https://i.imgur.com/cVAg5sA.png)

- Examples:
  - [Example with a UNet trained with thelper](examples/item.json)
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json)
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name       | Type                                        | Description                                                            |
|------------------|---------------------------------------------|------------------------------------------------------------------------|
| bands            | [Band Object](#bands)                       | Describes the EO data used to train or fine-tune the model.            |
| dlm:input        | [Input Object](#input-object)               | Describes the transformation between the EO data and the model input.  |
| dlm:architecture | [Architecture Object](#architecture-object) | Describes the model architecture.                                      |
| dlm:runtime      | [Runtime Object](#runtime-object)           | Describes the runtime environments to run the model (inference).       |
| dlm:output       | [Output Object](#output-object)             | Describes each model output and how to interpret it.                   |

In addition, fields from the following extensions must be imported in the item:
- [Scientific Extension Specification][stac-ext-sci] to describe relevant publications.
- [Version Extension Specification][stac-ext-ver] to define version tags.

[stac-ext-sci]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/scientific/README.md
[stac-ext-ver]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/version/README.md


### Bands

We use the STAC 1.1 Bands Object for representing bands information, including nodata value, data type, and common band names. Only bands used to train or fine tune the model should be included in this list.

### Input Object

| Field Name              | Type                            | Description                                                                                                                     |
|-------------------------|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                          | Python name of the input variable.                                                                                              |
| input_tensors           | [Tensor Object](#tensor-object) | Shape of the input tensor ($N \times C \times H \times W$).                                                                     |
| scaling_factor          | number                          | Scaling factor to apply to get data within `[0,1]`. For instance `scaling_factor=0.004` for 8-bit data.                         |
| mean                    | list of numbers                 | Mean vector value to be removed from the data if norm_type uses mean. The vector size must be consistent with `input_tensors:dim`. |
| std                     | list of numbers                 | Standard deviation values used to normalize the data if norm type uses standard deviation. The vector size must be consistent with `input_tensors:dim`. |
| band_names              | list of common metadata band names | Specifies the ordering of the bands selected from the bands list described in [bands](#Bands).                                 |


#### Tensor Object

| Field Name | Type   | Description                         |
|------------|--------|-------------------------------------|
| batch      | number | Batch size dimension (must be > 0). |
| time       | number | Number of timesteps  (must be > 0). |
| channels   | number | Number of channels  (must be > 0).  |
| height     | number | Height of the tensor (must be > 0). |
| width      | number | Width of the tensor (must be > 0).  |


### Architecture Object

| Field Name              | Type    | Description                                                 |
|-------------------------|---------|-------------------------------------------------------------|
| total_parameters        | integer | Total number of parameters.                                 |
| on_disk_size_mb         | number  | The equivalent memory size on disk in MB.                   |
| ram_size_mb | number    | number  | The equivalent memory size in memory in MB.                 |
| type                    | string  | Type of network (ex: ResNet-18).                            |
| summary                 | string  | Summary of the layers, can be the output of `print(model)`. |
| pretrained              | string  | Indicates the source of the pretraining (ex: ImageNet).     |

### Runtime Object

| Field Name            | Type                               | Description                                                                              |
|-----------------------|------------------------------------|------------------------------------------------------------------------------------------|
| framework             | string                             | Used framework (ex: PyTorch, TensorFlow).                                                |
| version               | string                             | Framework version (some models require a specific version of the framework).             |
| model_artifact        | string                             | Blob storage URI, POSIX filepath in docker image, or other URI type to the model file.   |
| model_handler         | string                             | Inference execution function.                                                            |
| model_src_url         | string                             | Url of the source code (ex: GitHub repo).                                                |
| model_commit_hash     | string                             | Hash value pointing to a specific version of the code.                                   |
| docker                | \[[Docker Object](#docker-object)] | Information for the deployment of the model in a docker instance.                        |
| batch_size_suggestion | number                             | A suggested batch size for a given compute instance type                                 |
| instance_suggestion   | str

#### Docker Object

| Field Name  | Type    | Description                                           |
|-------------|---------|-------------------------------------------------------|
| docker_file | string  | Url of the Dockerfile.                                |
| image_name  | string  | Name of the docker image.                             |
| tag         | string  | Tag of the image.                                     |
| working_dir | string  | Working directory in the instance that can be mapped. |
| run         | string  | Running command.                                      |
| accelerator         | boolean | True if the docker image requires a custom accelerator (CPU,TPU,MPS).              |

### Output Object

| Field Name               | Type                    | Description                                                                                                                                                                        |
|--------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task                     | [Task Enum](#task-enum) | Specifies the Machine Learning task for which the output can be used for.                                                                                                          |
| number_of_classes        | integer                 | Number of classes.                                                                                                                                                                 |
| final_layer_size         | \[integer]              | Sizes of the output tensor as ($N \times C \times H \times W$).                                                                                                                    |
| class_name_mapping       | list                    | Mapping of the output index to a short class name, for each record we specify the index and the class name.                                                                        |


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
