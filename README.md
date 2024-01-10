# Machine Learning Model Extension Specification

[![hackmd-github-sync-badge](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q/badge)](https://hackmd.io/3fB1lrHSTcSHQS57UhVk-Q)

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

The main objective of the extension is two-fold: 1) to enable building model collections that can be searched alongside associated STAC datasets and 2) to record all necessary bands, parameters, modeling artifact locations, and high-level processing steps to deploy an inference service. Specifically, this extension records the following information to make ML models searchable and reusable:
1. Sensor band specifications
2. Model input transforms including rescale and normalization
3. Model output shape, data type, and its semantic interpretation
4. An optional, flexible description of the runtime environment to be able to run the model
5. Scientific references

Note: The MLM specification is biased towards supervised ML models the produce classifications. However, fields that relate to supervised ML are optional and users can use the fields they need for different tasks.

Check the original technical report for an earlier version of the Model Extension [here](https://github.com/crim-ca/CCCOT03/raw/main/CCCOT03_Rapport%20Final_FINAL_EN.pdf) for more details.

![Image Description](https://i.imgur.com/cVAg5sA.png)

- Examples:
  - [Example with a ??? trained with torchgeo](examples/item.json) TODO update example
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json)
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name       | Type                                        | Description                                                                                                                                     |
|------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlm:input        | [[Model Input Object](#model-input-object)] | **REQUIRED.** Describes the transformation between the EO data and the model input.                                                             |
| mlm:architecture | [Architecture Object](#architecture-object) | **REQUIRED.** Describes the model architecture.                                                                                                 |
| mlm:runtime      | [Runtime Object](#runtime-object)           | **REQUIRED.** Describes the runtime environments to run the model (inference).                                                                  |
| mlm:output       | [Model Output Object](#model-output-object) | **REQUIRED.** Describes each model output and how to interpret it.                                                                              |
| parameters       | [Parameters Object](#params-object)         | Mapping with names for the parameters and their values. Some models may take additional scalars, tuples, and other non-tensor inputs like text. |

In addition, fields from the following extensions must be imported in the item:
- [Scientific Extension Specification][stac-ext-sci] to describe relevant publications.
- [Version Extension Specification][stac-ext-ver] to define version tags.

[stac-ext-sci]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/scientific/README.md
[stac-ext-ver]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/version/README.md

### Model Input Object

| Field Name              | Type                                          | Description                                                                                                                                                                                                                                        |
|-------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                                        | **REQUIRED.** Informative name of the input variable. Example "RGB Time Series"                                                                                                                                                                    |
| bands                   | [string]                                      | **REQUIRED.** Describes the EO bands used to train or fine-tune the model, which may be all or a subset of bands available in a STAC Item's [Band Object](#bands-and-statistics).                                                                  |
| input_feature           | [Feature Array Object](#feature-array-object) | **REQUIRED.** The N-dimensional feature array object that describes the shape, dimension ordering, and data type.                                                                                                                                  |
| parameters              | [Parameters Object](#params-object)           | Mapping with names for the parameters and their values. Some models may take additional scalars, tuples, and other non-tensor inputs like text.                                                                                                    |
| norm_by_channel         | boolean                                       | Whether to normalize each channel by channel-wise statistics or to normalize by dataset statistics.                                                                                                                                                |
| norm_type               | string                                        | Normalization method. Select one option from "min_max", "z_score", "max_norm", "mean_norm", "unit_variance", "none"                                                                                                                                |
| rescale_type            | string                                        | High-level descriptor of the rescaling method to change image shape. Select one option from "crop", "pad", "interpolation", "none". If your rescaling method combines more than one of these operations, provide the name of the operation instead |
| statistics              | [Statistics Object](stac-statistics)          | Dataset statistics for the training dataset used to normalize the inputs.                                                                                                                                                                          |
| pre_processing_function | string                                        | A url to the preprocessing function where normalization and rescaling takes place, and any other significant operations. Or, instead, the function code path, for example: my_python_module_name:my_processing_function                            |

#### Parameters Object

| Field Name                        | Type    | Description                                                              |
|-----------------------------------|---------|--------------------------------------------------------------------------|
| *parameter names depend on the model* | number | string | boolean | array  | The field number and names depend on the model as do the values. Values should be not be n-dimensional array inputs. If the model input can be represented as an n-dimensional array, it should instead be supplied as another model input object. |

The parameters field can either be specified in the model input object if they are associated with a specific input or as an Item or Collection field if the parameters are supplied without relation to a specific model input.

#### Bands and Statistics

We use the [STAC 1.1 Bands Object](https://github.com/radiantearth/stac-spec/pull/1254) for representing bands information, including nodata value, data type, and common band names. Only bands used to train or fine tune the model should be included in this `bands` field.

A deviation from the [STAC 1.1 Bands Object](https://github.com/radiantearth/stac-spec/pull/1254) is that we do not include the [Statistics](stac-statistics) object at the band object level, but at the Model Input level. This is because in machine learning, we typically only need statistics for the dataset used to train the model in order to normalize any given bands input.

[stac-statistics]: https://github.com/radiantearth/stac-spec/pull/1254/files#diff-2477b726f8c5d5d1c8b391be056db325e6918e78a24b414ccd757c7fbd574079R294

#### Feature Array Object

| Field Name | Type      | Description                                                                                                                                                                                                                                      |
|------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | [integer] | **REQUIRED.** Shape of the input n-dimensional feature array ($N \times C \times H \times W$), including the batch size dimension. The batch size dimension must either be greater than 0 or -1 to indicate an unspecified batch dimension size. |
| dim_order  | string    | **REQUIRED.** How the above dimensions are ordered with the tensor. "bhw", "bchw", "bthw", "btchw" are valid orderings where b=batch, c=channel, t=time, h=height, w=width                                                                       |
| dtype      | string    | **REQUIRED.** The data type of values in the feature array. Suggested to use [Numpy numerical types](https://numpy.org/devdocs/user/basics.types.html), omitting the numpy module, e.g. "float32"                                                |

### Architecture Object

| Field Name        | Type    | Description                                                                                   |
|-------------------|---------|-----------------------------------------------------------------------------------------------|
| name              | string  | **REQUIRED.** The name of the model architecture. For example, "ResNet-18" or "Random Forest" |
| file_size         | integer | **REQUIRED.** The size on disk of the model artifact (bytes).                                 |
| memory_size       | integer | **REQUIRED.** The in-memory size of the model on the accelerator during inference (bytes).    |
| summary           | string  | Summary of the layers, can be the output of `print(model)`.                                   |
| pretrained_source | string  | Indicates the source of the pretraining (ex: ImageNet).                                       |
| total_parameters  | integer | Total number of parameters.

### Runtime Object

| Field Name              | Type                                  | Description                                                                                                                                                                             |
|-------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| framework               | string                                | **REQUIRED.** Used framework (ex: PyTorch, TensorFlow).                                                                                                                                 |
| version                 | string                                | **REQUIRED.** Framework version (some models require a specific version of the framework).                                                                                              |
| model_asset             | [Asset Object](stac-asset)            | **REQUIRED.** Asset object containing URI to the model file.                                                                                                                            |
| source_code             | [Asset Object](stac-asset)            | **REQUIRED.** Source code description. Can describe a github repo, zip archive, etc. This description should reference the inference function, for example my_package.my_module.predict |
| accelerator             | [Accelerator Enum](#accelerator-enum) | **REQUIRED.** The intended accelerator that runs inference.                                                                                                                             |
| accelerator_constrained | boolean                               | **REQUIRED.** If the intended accelerator is the only accelerator that can run inference. If False, other accelerators, such as the amd64 (CPU), can run inference                      |
| hardware_summary        | string                                | **REQUIRED.** A high level description of the number of accelerators, specific generation of accelerator, or other relevant inference details.                                          |
| docker                  | [Container](#container)               | **RECOMMENDED.** Information for the deployment of the model in a docker instance.                                                                                                      |
| model_commit_hash       | string                                | Hash value pointing to a specific version of the code.                                                                                                                                  |
| batch_size_suggestion   | number                                | A suggested batch size for the accelerator and summarized hardware.                                                                                                                     |

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

| Field Name               | Type                                  | Description                                                                                                                                                                                                             |
|--------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task                     | [Task Enum](#task-enum)               | **REQUIRED.** Specifies the Machine Learning task for which the output can be used for.                                                                                                                                 |
| number_of_classes        | integer                               | Number of classes.                                                                                                                                                                                                      |
| result                   | [[Result Object](#result-object)]     | The list of output array/tensor from the model. For example ($N \times H \times W$). Use -1 to indicate variable dimensions, like the batch dimension.                                                                  |
| class_name_mapping       | [Class Map Object](#class-map-object) | Mapping of the class name to an index representing the label in the model output.                                                                                                                                       |
| post_processing_function | string                                | A url to the postprocessing function where normalization and rescaling takes place, and any other significant operations. Or, instead, the function code path, for example: my_package.my_module.my_processing_function |

While only `task` is a required field, all fields are recommended for supervised tasks that produce a fixed shape tensor and have output classes.
`image-captioning`, `multi-modal`, and `generative` tasks may not return fixed shape tensors or classes.

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

#### Result Object

| Field Name | Type      | Description                                                                                                                                                                                                                             |
|------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | [integer] | **REQUIRED.** Shape of the n-dimensional result array ($N \times H \times W$), possibly including a batch size dimension. The batch size dimension must either be greater than 0 or -1 to indicate an unspecified batch dimension size. |
| dim_names  | [string]  | **REQUIRED.** The names of the above dimensions of the result array.                                                                                                                                                                    |
| dtype      | string    | **REQUIRED.** The data type of values in the array. Suggested to use [Numpy numerical types](https://numpy.org/devdocs/user/basics.types.html), omitting the numpy module, e.g. "float32"                                               |


#### Class Map Object

| Field Name                        | Type    | Description                                                              |
|-----------------------------------|---------|--------------------------------------------------------------------------|
| *class names depend on the model* | integer | There are N corresponding integer values corresponding to N class fieds. |

The user can supply any number of fields for the classes of their model if the model produces a supervised classification result.                                                                                                                                                                   |

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
