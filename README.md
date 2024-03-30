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

The STAC Machine Learning Model (MLM) Extension provides a standard set of fields to describe machine learning models
trained on overhead imagery and enable running model inference.

The main objectives of the extension are:

1) to enable building model collections that can be searched alongside associated STAC datasets
2) record all necessary bands, parameters, modeling artifact locations, and high-level processing steps to deploy an inference service.

Specifically, this extension records the following information to make ML models searchable and reusable:
1. Sensor band specifications
2. Model input transforms including resize and normalization
3. Model output shape, data type, and its semantic interpretation
4. An optional, flexible description of the runtime environment to be able to run the model
5. Scientific references

The MLM specification is biased towards providing metadata fields for supervised machine learning models.
However, fields that relate to supervised ML are optional and users can use the fields they need for different tasks.

See [Best Practices](./best-practices.md) for guidance on what other STAC extensions you should use in conjunction with this extension.
The Machine Learning Model Extension purposely omits and delegates some definitions to other STAC extensions to favor
reusability and avoid metadata duplication whenever possible. A properly defined MLM STAC Item/Collection should almost
never have the Machine Learning Model Extension exclusively in `stac_extensions`.

Check the original [Technical Report](https://github.com/crim-ca/CCCOT03/raw/main/CCCOT03_Rapport%20Final_FINAL_EN.pdf)
for an earlier version of the MLM Extension, formerly known as the Deep Learning Model Extension (DLM). 
DLM was renamed to the current MLM Extension and refactored to form a cohesive definition across all machine
learning approaches, regardless of whether the approach constitutes a deep neural network or other statistical approach.
It also combines multiple definitions from the predecessor [ML-Model](https://github.com/stac-extensions/ml-model)
extension to synthesize common use cases into a single reference for Machine Learning Models.

![Image Description](https://i.imgur.com/cVAg5sA.png)

- Examples:
  - [Example with a ??? trained with torchgeo](examples/item.json) TODO update example
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json) TODO update
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

| Field Name                  | Type                                             | Description                                                                                                                                                                                                                                                                                 |
|-----------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlm:name                    | string                                           | **REQUIRED** A unique name for the model. This can include, but must be distinct, from simply naming the model architecture. If there is a publication or other published work related to the model, use the official name of the model.                                                    |
| mlm:architecture            | [Model Architecture](#model-architecture) string | **REQUIRED** A generic and well established architecture name of the model.                                                                                                                                                                                                                 | 
| mlm:tasks                   | [[Task Enum](#task-enum)]                        | **REQUIRED** Specifies the Machine Learning tasks for which the model can be used for. If multi-tasks outputs are provided by distinct model heads, specify all available tasks under the main properties and specify respective tasks in each [Model Output Object](#model-output-object). |
| mlm:framework               | string                                           | **REQUIRED** Framework used to train the model (ex: PyTorch, TensorFlow).                                                                                                                                                                                                                   |
| mlm:framework_version       | string                                           | The `framework` library version. Some models require a specific version of the machine learning `framework` to run.                                                                                                                                                                         |
| mlm:memory_size             | integer                                          | **REQUIRED** The in-memory size of the model on the accelerator during inference (bytes).                                                                                                                                                                                                   |
| mlm:accelerator             | [Accelerator Enum](#accelerator-enum) \| null    | The intended computational hardware that runs inference. If undefined or set to `null` explicitly, the model does not require any specific accelerator.                                                                                                                                     |
| mlm:accelerator_constrained | boolean                                          | Indicates if the intended `accelerator` is the only `accelerator` that can run inference. If undefined, it should be assumed `false`.                                                                                                                                                       |
| mlm:accelerator_summary     | string                                           | A high level description of the `accelerator`, such as its specific generation, or other relevant inference details.                                                                                                                                                                        |
| mlm:accelerator_count       | integer                                          | A minimum amount of `accelerator` instances required to run the model.                                                                                                                                                                                                                      | 
| mlm:total_parameters        | integer                                          | Total number of model parameters, including trainable and non-trainable parameters.                                                                                                                                                                                                         |
| mlm:pretrained_source       | string \| null                                   | The source of the pretraining. Can refer to popular pretraining datasets by name (i.e. Imagenet) or less known datasets by URL and description. If trained from scratch, the `null` value should be set explicitly.                                                                         |
| mlm:batch_size_suggestion   | number                                           | A suggested batch size for the accelerator and summarized hardware.                                                                                                                                                                                                                         |
| mlm:input                   | [[Model Input Object](#model-input-object)]      | **REQUIRED** Describes the transformation between the EO data and the model input.                                                                                                                                                                                                          |
| mlm:output                  | [[Model Output Object](#model-output-object)]    | **REQUIRED** Describes each model output and how to interpret it.                                                                                                                                                                                                                           |

In addition, fields from the following extensions must be imported in the item:
- [Scientific Extension Specification][stac-ext-sci] to describe relevant publications.
- [Version Extension Specification][stac-ext-ver] to define version tags.

[stac-ext-sci]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/scientific/README.md
[stac-ext-ver]: https://github.com/radiantearth/stac-spec/tree/v1.0.0-beta.2/extensions/version/README.md

### Model Architecture

In most cases, this should correspond to common architecture names defined in the literature,
such as `ResNet`, `VGG`, `GAN` or `Vision Transformer`. For more examples of proper names (including casing),
the [Papers With Code - Computer Vision Methods](https://paperswithcode.com/methods/area/computer-vision) can be used.
Note that this field is not an explicit "Enum", and is used only as an indicator of common architecture occurrences.
If no specific or predefined architecture can be associated with the described model, simply employ `unknown` or
another custom name as deemed appropriate.

### Task Enum

It is recommended to define `mlm:tasks` of the entire model at the STAC Item level,
and `tasks` of respective [Model Output Object](#model-output-object) with the following values.
Although other values are permitted to support more use cases, they should be used sparingly to allow better
interoperability of models and their representation.

As a general rule of thumb, if a task is not represented below, an appropriate name can be formulated by taking
definitions listed in [Papers With Code](https://paperswithcode.com/sota). The names
should be normalized to lowercase and use hyphens instead of spaces.

| Task Name               | Corresponding `label:tasks` | Description                                                                                                     |
|-------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------|
| `regression`            | `regression`                | Generic regression that estimates a numeric and continuous value.                                               |
| `classification`        | `classification`            | Generic classification task that assigns class labels to an output.                                             |
| `scene-classification`  | *n/a*                       | Specific classification task where the model assigns a single class label to an entire scene/area.              |
| `detection`             | `detection`                 | Generic detection of the "presence" of objects or entities, with or without positions.                          |
| `object-detection`      | *n/a*                       | Task corresponding to the identification of positions as bounding boxes of object detected in the scene.        |
| `segmentation`          | `segmentation`              | Generic tasks that regroups all types of segmentations tasks consisting of applying labels to pixels.           |
| `semantic-segmentation` | *n/a*                       | Specific segmentation task where all pixels are attributed labels, without consideration of similar instances.  |
| `instance-segmentation` | *n/a*                       | Specific segmentation task that assigns distinct labels for groups of pixels corresponding to object instances. |
| `panoptic-segmentation` | *n/a*                       | Specific segmentation task that combines instance segmentation of objects and semantic labels for non-objects.  |
| `similarity-search`     | *n/a*                       | Generic task to identify whether a query input corresponds to another reference within a corpus.                |
| `image-captioning`      | *n/a*                       | Specific task of describing the content of an image in words.                                                   |
| `generative`            | *n/a*                       | Generic task that encompasses all synthetic data generation techniques.                                         |
| `super-resolution`      | *n/a*                       | Specific task that increases the quality and resolution of an image by increasing its high-frequency details.   |

If the task falls within the category of supervised machine learning and uses labels during training,
this should align with the `label:tasks` values defined in [STAC Label Extension][stac-ext-label-props] for relevant
STAC Collections and Items published with the model described by this extension.

It is to be noted that multiple "*generic*" tasks names (`classification`, `detection`, etc.) are defined to allow
correspondance with `label:tasks`, but these can lead to some ambiguity depending on context. For example, a model
that supports `classification` could mean that the model can predict patch-based classes over an entire scene
(i.e.: `scene-classification` for a single prediction over an entire area of interest as a whole),
or that it can predict pixel-wise "classifications", such as land-cover labels for
every single pixel coordinate over the area of interest. Maybe counter-intuitively to some users,
such a model that produces pixel-wise "classifications" should be attributed the `segmentation` task
(and more specifically `semantic-segmentation`) rather than `classification`. To avoid this kind of ambiguity,
it is strongly recommended that `tasks` always aim to provide the most specific definitions possible to explicitly
describe what the model accomplishes.

[stac-ext-label-props]: https://github.com/stac-extensions/label#item-properties

### Accelerator Type Enum

It is recommended to define `accelerator` with one of the following values:

- `amd64` models compatible with AMD or Intel CPUs (no hardware specific optimizations)
- `cuda` models compatible with NVIDIA GPUs
- `xla` models compiled with XLA. Models trained on TPUs are typically compiled with XLA.
- `amd-rocm` models trained on AMD GPUs
- `intel-ipex-cpu` for models optimized with IPEX for Intel CPUs
- `intel-ipex-gpu` for models optimized with IPEX for Intel GPUs
- `macos-arm` for models trained on Apple Silicon

> [!WARNING]
> If `mlm:accelerator = amd64`, this explicitly indicates that the model does not (and will not try to) use any
> accelerator, even if some are available from the runtime environment. This is to be distinguished from 
> the value `mlm:accelerator = null`, which means that the model *could* make use of some accelerators if provided,
> but is not constrained by any specific one. To improve comprehension by users, it is recommended that any model
> using `mlm:accelerator = amd64` also set explicitly `mlm:accelerator_constrained = true` to illustrate that the
> model **WILL NOT** use accelerators, although the hardware resolution should be identical nonetheless.

When `mlm:accelerator = null` is employed, the value of `mlm:accelerator_constrained` can be ignored, since even if
set to `true`, there would be no `accelerator` to contain against. To avoid confusion, it is suggested to set the
`mlm:accelerator_constrained = false` or omit the field entirely in this case.

### Model Input Object

| Field Name              | Type                                              | Description                                                                                                                                                                                                                                                                   |
|-------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                                            | **REQUIRED** Name of the input variable defined by the model. If no explicit name is defined by the model, an informative name (e.g.: "RGB Time Series") can be used instead.                                                                                                 | 
| bands                   | [string]                                          | **REQUIRED** The names of the raster bands used to train or fine-tune the model, which may be all or a subset of bands available in a STAC Item's [Band Object](#bands-and-statistics).                                                                                       |
| input                   | [Input Structure Object](#input-structure-object) | **REQUIRED** The N-dimensional array definition that describes the shape, dimension ordering, and data type.                                                                                                                                                                  |
| norm_by_channel         | boolean                                           | Whether to normalize each channel by channel-wise statistics or to normalize by dataset statistics. If True, use an array of `statistics` of same dimensionality and order as the `bands` field in this object.                                                               |
| norm_type               | string \| null                                    | Normalization method. Select one option from `"min_max"`, `"z_score"`, `"max_norm"`, `"mean_norm"`, `"unit_variance"`, `"norm_with_clip"` or `null` when none applies.                                                                                                        |
| resize_type             | string \| null                                    | High-level descriptor of the rescaling method to change image shape. Select one option from `"crop"`, `"pad"`, `"interpolation"` or `null` when none applies. If your rescaling method combines more than one of these operations, provide the name of the operation instead. |
| statistics              | [[Statistics Object](#bands-and-statistics)]      | Dataset statistics for the training dataset used to normalize the inputs.                                                                                                                                                                                                     |
| norm_with_clip_values   | [integer]                                         | If `norm_type = "norm_with_clip"` this array supplies a value that is less than the band maximum. The array must be the same length as `bands`, each value is used to divide each band before clipping values between 0 and 1.                                                |
| pre_processing_function | string \| null                                    | URI to the preprocessing function where normalization and rescaling takes place, and any other significant operations or, instead, the function code path, for example: `my_python_module_name:my_processing_function`.                                                       |

Fields that accept the `null` value can be considered `null` when omitted entirely for parsing purposes.
However, setting `null` explicitly when this information is known by the model provider can help users understand
what is the expected behavior of the model. It is therefore recommended to provide `null` explicitly when applicable.

## Assets Objects

| Field Name      | Type                       | Description                                                                               |
|-----------------|----------------------------|-------------------------------------------------------------------------------------------|
| mlm:model       | [Asset Object][stac-asset] | **REQUIRED** Asset object containing the model definition.                                |
| mlm:source_code | [Asset Object][stac-asset] | **RECOMMENDED** Source code description. Can describe a Git repository, ZIP archive, etc. |
| mlm:container   | [Asset Object][stac-asset] | **RECOMMENDED** Information to run the model in a container with URI to the container.    |
| mlm:training    | [Asset Object][stac-asset] | **RECOMMENDED** Information to run the training pipeline of the model being described.    |
| mlm:inference   | [Asset Object][stac-asset] | **RECOMMENDED** Information to run the inference pipeline of the model being described.   |

It is recommended that the [Assets][stac-asset] defined in a STAC Item using MLM extension use the above field property 
names for nesting the Assets in order to improve their quick identification, although the specific names employed are
left up to user preference. However, the MLM Asset definitions **MUST** include the
appropriate [MLM Asset Roles](#mlm-asset-roles) to ensure their discovery.

[stac-asset]: https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#asset-object

### MLM Asset Roles

Asset `roles` should include relevant names that describe them. This does not only include 
the [Recommended Asset Roles](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#asset-roles)
from the core specification, such as `data` or `metadata`, but also descriptors such as `mlm:model`, `mlm:weights` and
so on, as applicable for the relevant [MLM Asset](#mlm-assets) being described. Please refer to the following sections
for `roles` requirements by specific [MLM Asset](#mlm-assets).

Note that `mlm:` prefixed roles are used for identification purpose of the Assets, but non-prefixed roles can be
provided as well to offer generic descriptors. For example, `["mlm:model", "model", "data"]` could be considered for
the [Model Asset](#model-asset).

In order to provide more context, the following roles are also recommended were applicable:

| Asset Role                | Additional Roles        | Description                                                                              |
|---------------------------|-------------------------|------------------------------------------------------------------------------------------|
| mlm:inference-runtime (*) | `runtime`               | Describes an Asset that provides runtime reference to perform model inference.           |
| mlm:training-runtime (*)  | `runtime`               | Describes an Asset that provides runtime reference to perform model training.            |
| mlm:checkpoint (*)        | `weights`, `checkpoint` | Describes an Asset that provides a model checkpoint with embedded model configurations.  |
| mlm:weights               | `weights`, `checkpoint` | Describes an Asset that provides a model weights (typically some Tensor representation). |
| mlm:model                 | `model`                 | Required role for [Model Asset](#model-asset).                                           |
| mlm:source_code           | `code`                  | Required role for [Model Asset](#source-code-asset).                                     |

> [!NOTE]
> (*) These roles are offered as direct conversions from the previous extension
> that provided [ML-Model Asset Roles][ml-model-asset-roles] to provide easier upgrade to the MLM extension.

[ml-model-asset-roles]: https://github.com/stac-extensions/ml-model?tab=readme-ov-file#asset-objects


### Model Asset

| Field Name        | Type                                      | Description                                                                                      |
|-------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------|
| title             | string                                    | Description of the model asset.                                                                  |
| href              | string                                    | URI to the model artifact.                                                                       |
| type              | string                                    | The media type of the artifact (see [Model Artifact Media-Type](#model-artifact-media-type).     |
| roles             | [string]                                  | **REQUIRED** Specify `mlm:model`. Can include `["mlm:weights", "mlm:checkpoint"]` as applicable. |
| mlm:artifact_type | [Artifact Type Enum](#artifact-type-enum) | Specifies the kind of model artifact. Typically related to a particular ML framework.            |

Recommended Asset `roles` include `mlm:weights` or `mlm:checkpoint` for model weights that need to be loaded by a
model definition and `mlm:compiled` for models that can be loaded directly without an intermediate model definition.
In each case, the `mlm:model` should be applied as well to indicate that this asset represents the model.

It is also recommended to make use of the
[file](https://github.com/stac-extensions/file?tab=readme-ov-file#asset--link-object-fields)
extension for this Asset, as it can provide useful information to validate the contents of the model definition,
by comparison with fields `file:checksum` and `file:size` for example.

#### Model Artifact Media-Type

Not all ML framework, libraries or model artifacts provide explicit media-type. When those are not provided, custom
media-types can be considered. For example `application/x-pytorch` or `application/octet-stream; application=pytorch`
could be appropriate to represent a PyTorch `.pt` file, since the underlying format is a serialized pickle structure.

#### Artifact Type Enum

This value can be used to provide additional details about the specific model artifact being described.
For example, PyTorch offers various strategies for providing model definitions, such as Pickle (`.pt`), TorchScript,
or the compiled approach. Since they all refer to the same ML framework,
the [Model Artifact Media-Type](#model-artifact-media-type) would be insufficient in this case to detect with strategy
should be used. 

Following are some proposed *Artifact Type* values for corresponding approaches, but other names are
permitted as well. Note that the names are selected using the framework-specific definitions to help
the users understand the source explicitly, although this is not strictly required either.

| Artifact Type      | Description                                                                                                              |
|--------------------|--------------------------------------------------------------------------------------------------------------------------|
| `torch.compile`    | A model artifact obtained by [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).  |
| `torch.jit.script` | A model artifact obtained by [`TorchScript`](https://pytorch.org/docs/stable/jit.html).                                  |
| `torch.save`       | A model artifact saved by [Serialized Pickle Object](https://pytorch.org/tutorials/beginner/saving_loading_models.html). |

### Source Code Asset

| Field Name     | Type     | Description                                                                   |
|----------------|----------|-------------------------------------------------------------------------------|
| title          | string   | Title of the source code.                                                     |
| href           | string   | URI to the code repository, a ZIP archive, or an individual code/script file. |
| type           | string   | Media-type of the URI.                                                        |
| roles          | [string] | **RECOMMENDED** Specify one or more of `["model", "code", "metadata"]`        |
| description    | string   | Description of the source code.                                               |
| mlm:entrypoint | string   | Specific entrypoint reference in the code to use for running model inference. |

If the referenced code does not directly offer a callable script to run the model, the `mlm:entrypoint` field should be
added to the [Asset Object][stac-asset] in order to provide a pointer to the inference function to execute the model.
For example, `my_package.my_module:predict` would refer to the `predict` function located in the `my_module` inside the
`my_package` library provided by the repository.

It is strongly recommended to use a specific media-type such as `text/x-python` if the source code refers directly
to a script of a known programming language. Using the HTML rendering of that source file, such as though GitHub
for example, should be avoided. Using the "Raw Contents" endpoint for such cases is preferable.
The `text/html` media-type should be reserved for cases where the URI generally points at a Git repository.
Note that the URI including the specific commit hash, release number or target branch should be preferred over
other means of referring to checkout procedures, although this specification does not prohibit the use of additional
properties to better describe the Asset.

Since the source code of a model provides useful example on how to use it, it is also recommended to define relevant
references to documentation using the `example` extension.
See the [Best Practices - Example Extension](best-practices.md#example-extension) section for more details.

Recommended asset `roles` include `code` and `metadata`,
since the source code asset might also refer to more detailed metadata than this specification captures.

### Container Asset

| Field Name  | Type     | Description                                                                       |
|-------------|----------|-----------------------------------------------------------------------------------|
| title       | string   | Description of the container.                                                     |
| href        | string   | URI of the published container, including the container registry, image and tag.  |
| type        | string   | Media-type of the container, typically `application/vnd.oci.image.index.v1+json`. |
| roles       | [string] | Specify `["runtime"]` and any other custom roles.                                 |

If you're unsure how to containerize your model, we suggest starting from the latest official container image for
your framework that works with your model and pinning the container version.

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


#### Bands and Statistics

Depending on the supported `stac_version` and other `stac_extensions` employed by the STAC Item using MLM,
the [STAC 1.1 - Band Object][stac-1.1-band], 
the [STAC Raster - Band Object][stac-raster-band] or
the [STAC EO - Band Object][stac-eo-band] can be used for
representing bands information, including notably the `nodata` value,
the `data_type` (see also [Data Type Enum](#data-type-enum)),
and [Common Band Names][stac-band-names].

Only bands used as input to the model should be included in the MLM `bands` field.
To avoid duplicating the information, MLM only uses the `name` of whichever "Band Object" is defined in the STAC Item.

One distinction from the [STAC 1.1 - Band Object][stac-1.1-band] in MLM is that [Statistics][stac-1.1-stats] object
(or the corresponding [STAC Raster - Statistics][stac-raster-stats] for STAC 1.0) are not
defined at the "Band Object" level, but at the [Model Input](#model-input-object) level.
This is because, in machine learning, it is common to need overall statistics for the dataset used to train the model
to normalize all bands, rather than normalizing the values over a single product. Furthermore, statistics could be
applied differently for distinct [Model Input](#model-input-object) definitions, in order to adjust for intrinsic
properties of the model.

[stac-1.1-band]: https://github.com/radiantearth/stac-spec/pull/1254
[stac-1.1-stats]: https://github.com/radiantearth/stac-spec/pull/1254/files#diff-2477b726f8c5d5d1c8b391be056db325e6918e78a24b414ccd757c7fbd574079R294
[stac-eo-band]: https://github.com/stac-extensions/eo?tab=readme-ov-file#band-object
[stac-raster-band]: https://github.com/stac-extensions/raster?tab=readme-ov-file#raster-band-object
[stac-raster-stats]: https://github.com/stac-extensions/raster?tab=readme-ov-file#statistics-object
[stac-band-names]: https://github.com/stac-extensions/eo?tab=readme-ov-file#common-band-names

#### Data Type Enum

When describing the `data_type` provided by a [Band](#bands-and-statistics), whether for defining
the [Input Structure](#input-structure-object) or the [Result Structure](#result-structure-object),
the [Data Types from the STAC Raster extension][raster-data-types] should be used.

[raster-data-types]: https://github.com/stac-extensions/raster?tab=readme-ov-file#data-types

#### Input Structure Object

| Field Name | Type                              | Description                                                                                                                                                                                                             |
|------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | [integer]                         | **REQUIRED** Shape of the input n-dimensional array ($N \times C \times H \times W$), including the batch size dimension. Each dimension must either be greater than 0 or -1 to indicate a variable dimension size.     |
| dim_order  | string                            | **REQUIRED** How the above dimensions are ordered within the `shape`. `bhw`, `bchw`, `bthw`, `btchw` are valid orderings where `b`=batch, `c`=channel, `t`=time, `h`=height, `w`=width.                                 |
| data_type  | [Data Type Enum](#data-type-enum) | **REQUIRED** The data type of values in the n-dimensional array. For model inputs, this should be the data type of the processed input supplied to the model inference function, not the data type of the source bands. |

A common use of `-1` for one dimension of `shape` is to indicate a variable batch-size.
However, this value is not strictly reserved for the `b` dimension.
For example, if the model is capable of automatically adjusting its input layer to adapt to the provided input data,
then the corresponding dimensions that can be adapted can employ `-1` as well. 

### Model Output Object

| Field Name               | Type                                                | Description                                                                                                                                                                                             |
|--------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tasks                    | [[Task Enum](#task-enum)]                           | **REQUIRED** Specifies the Machine Learning tasks for which the output can be used for. This can be a subset of `mlm:tasks` defined under the Item `properties` as applicable.                          |
| result                   | [Result Structure Object](#result-structure-object) | The structure that describes the resulting output arrays/tensors from one model head.                                                                                                                   |
| classification:classes   | [[Class Object](#class-object)]                     | A list of class objects adhering to the [Classification Extension](https://github.com/stac-extensions/classification).                                                                                  |
| post_processing_function | string                                              | A url to the postprocessing function where normalization, rescaling, and other operations take place.. Or, instead, the function code path, for example: `my_package.my_module:my_processing_function`. |

While only `tasks` is a required field, all fields are recommended for tasks that produce a fixed
shape tensor and have output classes. Outputs that have variable dimensions, can define the `result` with the
appropriate dimension value `-1` in the `shape` field. When the model does not produce specific classes, such 
as for `regression`, `image-captioning`, `super-resolution` and some `generative` tasks, to name a few, the
`classification:classes` can be omitted.

#### Result Structure Object

| Field Name | Type                              | Description                                                                                                                                                                                                                            |
|------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | [integer]                         | **REQUIRED** Shape of the n-dimensional result array ($N \times H \times W$), possibly including a batch size dimension. The batch size dimension must either be greater than 0 or -1 to indicate an unspecified batch dimension size. |
| dim_names  | [string]                          | **REQUIRED** The names of the above dimensions of the result array, ordered the same as this object's `shape` field.                                                                                                                   |
| data_type  | [Data Type Enum](#data-type-enum) | **REQUIRED** The data type of values in the n-dimensional array. For model outputs, this should be the data type of the result of the model inference  without extra post processing.                                                  |

#### Class Object

See the documentation for the
[Class Object](https://github.com/stac-extensions/classification?tab=readme-ov-file#class-object).

## Relation types

The following types should be used as applicable `rel` types in the
[Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object)
of STAC Items describing Band Assets that result from the inference of a model described by the MLM extension.

| Type         | Description                                                                                                                                  |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| derived_from | This link points to a STAC Collection or Item using MLM, using the corresponding [`mlm:name`](#item-properties-and-collection-fields) value. |

Note that a derived product from model inference described by STAC should also consider using
additional indications that it came of a model, such as described by
the [Best Practices - Processing Extension](best-practices.md#processing-extension).

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
