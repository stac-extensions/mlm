# Machine Learning Model Extension Specification

[![hackmd-github-sync-badge](https://hackmd.io/N1cWyDM2S9eaAQtSvS0J_A/badge)](https://hackmd.io/N1cWyDM2S9eaAQtSvS0J_A?both)

- **Title:** Machine Learning Model Extension
- **Identifier:** [https://crim-ca.github.io/mlm-extension/v1.1.0/schema.json](https://crim-ca.github.io/mlm-extension/v1.1.0/schema.json)
- **Field Name Prefix:** mlm
- **Scope:** Collection, Item, Asset, Links
- **Extension Maturity Classification:** Pilot
- **Owner:**
  - [@fmigneault](https://github.com/fmigneault)
  - [@rbavery](https://github.com/rbavery)
  - [@ymoisan](https://github.com/ymoisan)
  - [@sfoucher](https://github.com/sfoucher)

The STAC Machine Learning Model (MLM) Extension provides a standard set of fields to describe machine learning models
trained on overhead imagery and enable running model inference.

The main objectives of the extension are:

1) to enable building model collections that can be searched alongside associated STAC datasets
2) record all necessary bands, parameters, modeling artifact locations, and high-level processing steps to deploy
   an inference service.

Specifically, this extension records the following information to make ML models searchable and reusable:
1. Sensor band specifications
2. Model input transforms including resize and normalization
3. Model output shape, data type, and its semantic interpretation
4. An optional, flexible description of the runtime environment to be able to run the model
5. Scientific references

The MLM specification is biased towards providing metadata fields for supervised machine learning models.
However, fields that relate to supervised ML are optional and users can use the fields they need for different tasks.

See [Best Practices](./best-practices.md) for guidance on what other STAC extensions you should use in conjunction
with this extension.
The Machine Learning Model Extension purposely omits and delegates some definitions to other STAC extensions to favor
reusability and avoid metadata duplication whenever possible. A properly defined MLM STAC Item/Collection should almost
never have the Machine Learning Model Extension exclusively in `stac_extensions`.

For details about the earlier (legacy) version of the MLM Extension, formerly known as
the *Deep Learning Model Extension* (DLM), please refer to the [DLM LEGACY](README_DLM_LEGACY.md) document.
DLM was renamed to the current MLM Extension and refactored to form a cohesive definition across all machine
learning approaches, regardless of whether the approach constitutes a deep neural network or other statistical approach.
It also combines multiple definitions from the predecessor [ML-Model](https://github.com/stac-extensions/ml-model)
extension to synthesize common use cases into a single reference for Machine Learning Models.

For more details about the [`stac-model`](stac_model) Python package, which provides definitions of the MLM extension
using both [`Pydantic`](https://docs.pydantic.dev/latest/) and [`PySTAC`](https://pystac.readthedocs.io/en/stable/)
connectors, please refer to the [STAC Model](README_STAC_MODEL.md) document.

> :warning: <br>
> FIXME: update examples 

- Examples:
  - **Example with a ??? trained with torchgeo** :warning: TODO update example
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection
- [JSON Schema](json-schema/schema.json) TODO update
- [Changelog](./CHANGELOG.md)

## Item Properties and Collection Fields

The fields in the table below can be used in these parts of STAC documents:

- [ ] Catalogs
- [x] Collections
- [x] Item Properties (incl. Summaries in Collections)
- [x] Assets (for both Collections and Items, incl. [Item-Assets][item-assets] definitions in Collections)
- [ ] Links

[item-assets]: https://github.com/stac-extensions/item-assets

| Field Name                  | Type                                                          | Description                                                                                                                                                                                                                                                                                 |
|-----------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlm:name                    | string                                                        | **REQUIRED** A name for the model. This can include, but must be distinct, from simply naming the model architecture. If there is a publication or other published work related to the model, use the official name of the model.                                                    |
| mlm:architecture            | [Model Architecture](#model-architecture) string              | **REQUIRED** A generic and well established architecture name of the model.                                                                                                                                                                                                                 | 
| mlm:tasks                   | \[[Task Enum](#task-enum)]                                    | **REQUIRED** Specifies the Machine Learning tasks for which the model can be used for. If multi-tasks outputs are provided by distinct model heads, specify all available tasks under the main properties and specify respective tasks in each [Model Output Object](#model-output-object). |
| mlm:framework               | string                                                        | Framework used to train the model (ex: PyTorch, TensorFlow).                                                                                                                                                                                                                   |
| mlm:framework_version       | string                                                        | The `framework` library version. Some models require a specific version of the machine learning `framework` to run.                                                                                                                                                                         |
| mlm:memory_size             | integer                                                       | The in-memory size of the model on the accelerator during inference (bytes).                                                                                                                                                                                                                |
| mlm:total_parameters        | integer                                                       | Total number of model parameters, including trainable and non-trainable parameters.                                                                                                                                                                                                         |
| mlm:pretrained              | boolean                                                       | Indicates if the model was pretrained. If the model was pretrained, consider providing `pretrained_source` if it is known.                                                                                                                                                                  |
| mlm:pretrained_source       | string \| null                                                | The source of the pretraining. Can refer to popular pretraining datasets by name (i.e. Imagenet) or less known datasets by URL and description. If trained from scratch (i.e.: `pretrained = false`), the `null` value should be set explicitly.                                            |
| mlm:batch_size_suggestion   | integer                                                       | A suggested batch size for the accelerator and summarized hardware.                                                                                                                                                                                                                         |
| mlm:accelerator             | [Accelerator Type Enum](#accelerator-type-enum) \| null       | The intended computational hardware that runs inference. If undefined or set to `null` explicitly, the model does not require any specific accelerator.                                                                                                                                     |
| mlm:accelerator_constrained | boolean                                                       | Indicates if the intended `accelerator` is the only `accelerator` that can run inference. If undefined, it should be assumed `false`.                                                                                                                                                       |
| mlm:accelerator_summary     | string                                                        | A high level description of the `accelerator`, such as its specific generation, or other relevant inference details.                                                                                                                                                                        |
| mlm:accelerator_count       | integer                                                       | A minimum amount of `accelerator` instances required to run the model.                                                                                                                                                                                                                      | 
| mlm:input                   | \[[Model Input Object](#model-input-object)]                  | **REQUIRED** Describes the transformation between the EO data and the model input.                                                                                                                                                                                                          |
| mlm:output                  | \[[Model Output Object](#model-output-object)]                | **REQUIRED** Describes each model output and how to interpret it.                                                                                                                                                                                                                           |
| mlm:hyperparameters         | [Model Hyperparameters Object](#model-hyperparameters-object) | Additional hyperparameters relevant for the model.                                                                                                                                                                                                                                          |

To decide whether above fields should be applied under Item `properties` or under respective Assets, the context of
each field must be considered. For example, the `mlm:name` should always be provided in the Item `properties`, since
it relates to the model as a whole. In contrast, some models could support multiple `mlm:accelerator`, which could be
handled by distinct source code represented by different Assets. In such case, `mlm:accelerator` definitions should be
nested under their relevant Asset. If a field is defined both at the Item and Asset level, the value at the Asset level
would be considered for that specific Asset, and the value at the Item level would be used for other Assets that did
not override it for their respective reference. For some of the fields, further details are provided in following
sections to provide more precisions regarding some potentially ambiguous use cases.

In addition, fields from the multiple relevant extensions should be defined as applicable. See
[Best Practices - Recommended Extensions to Compose with the ML Model Extension](best-practices.md#recommended-extensions-to-compose-with-the-ml-model-extension)
for more details.

For the [Extent Object][stac-extent]
in STAC Collections and the corresponding spatial and temporal fields in Items, please refer to section
[Best Practices - Using STAC Common Metadata Fields for the ML Model Extension][stac-mlm-meta].

[stac-mlm-meta]: best-practices.md#using-stac-common-metadata-fields-for-the-ml-model-extension
[stac-extent]: https://github.com/radiantearth/stac-spec/blob/master/collection-spec/collection-spec.md#extent-object

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

| Task Name               | Corresponding `label:tasks` | Description                                                                                                              |
|-------------------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `regression`            | `regression`                | Generic regression that estimates a numeric and continuous value.                                                        |
| `classification`        | `classification`            | Generic classification task that assigns class labels to an output.                                                      |
| `scene-classification`  | *n/a*                       | Specific classification task where the model assigns a single class label to an entire scene/area.                       |
| `detection`             | `detection`                 | Generic detection of the "presence" of objects or entities, with or without positions.                                   |
| `object-detection`      | *n/a*                       | Task corresponding to the identification of positions as bounding boxes of object detected in the scene.                 |
| `segmentation`          | `segmentation`              | Generic tasks that regroups all types of segmentations tasks consisting of applying labels to pixels.                    |
| `semantic-segmentation` | *n/a*                       | Specific segmentation task where all pixels are attributed labels, without consideration for segments as unique objects. |
| `instance-segmentation` | *n/a*                       | Specific segmentation task that assigns distinct labels for groups of pixels corresponding to object instances.          |
| `panoptic-segmentation` | *n/a*                       | Specific segmentation task that combines instance segmentation of objects and semantic labels for non-objects.           |
| `similarity-search`     | *n/a*                       | Generic task to identify whether a query input corresponds to another reference within a corpus.                         |
| `generative`            | *n/a*                       | Generic task that encompasses all synthetic data generation techniques.                                                  |
| `image-captioning`      | *n/a*                       | Specific task of describing the content of an image in words.                                                            |
| `super-resolution`      | *n/a*                       | Specific task that increases the quality and resolution of an image by increasing its high-frequency details.            |

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

### Framework

This should correspond to the common library name of a well-established ML framework.
No "Enum" are *enforced* to allow easy addition of newer frameworks, but it is **STRONGLY** recommended
to use common names when applicable. Below are a few notable entries.

- `PyTorch`
- `TensorFlow`
- `scikit-learn`
- `Hugging Face`
- `Keras`
- `ONNX`
- `rgee`
- `spatialRF`
- `JAX`
- `MXNet`
- `Caffe`
- `PyMC`
- `Weka`

### Accelerator Type Enum

It is recommended to define `accelerator` with one of the following values:

- `amd64` models compatible with AMD or Intel CPUs (no hardware specific optimizations)
- `cuda` models compatible with NVIDIA GPUs
- `xla` models compiled with XLA. Models trained on TPUs are typically compiled with XLA.
- `amd-rocm` models trained on AMD GPUs
- `intel-ipex-cpu` for models optimized with IPEX for Intel CPUs
- `intel-ipex-gpu` for models optimized with IPEX for Intel GPUs
- `macos-arm` for models trained on Apple Silicon

> :warning: <br>
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

| Field Name              | Type                                                    | Description                                                                                                                                                                                                                                   |
|-------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                                                  | **REQUIRED** Name of the input variable defined by the model. If no explicit name is defined by the model, an informative name (e.g.: `"RGB Time Series"`) can be used instead.                                                               | 
| bands                   | \[string]                                               | **REQUIRED** The names of the raster bands used to train or fine-tune the model, which may be all or a subset of bands available in a STAC Item's [Band Object](#bands-and-statistics). If no band applies for one input, use an empty array. |
| input                   | [Input Structure Object](#input-structure-object)       | **REQUIRED** The N-dimensional array definition that describes the shape, dimension ordering, and data type.                                                                                                                                  |
| description             | string                                                  | Additional details about the input such as describing its purpose or expected source that cannot be represented by other properties.                                                                                                          | 
| norm_by_channel         | boolean                                                 | Whether to normalize each channel by channel-wise statistics or to normalize by dataset statistics. If True, use an array of `statistics` of same dimensionality and order as the `bands` field in this object.                               |
| norm_type               | [Normalize Enum](#normalize-enum) \| null               | Normalization method. Select an appropriate option or `null` when none applies. Consider using `pre_processing_function` for custom implementations or more complex combinations.                                                             |
| norm_clip               | \[number]                                               | When `norm_type = "clip"`, this array supplies the value for each `bands` item, which is used to divide each band before clipping values between 0 and 1.                                                                                     |
| resize_type             | [Resize Enum](#resize-enum) \| null                     | High-level descriptor of the rescaling method to change image shape. Select an appropriate option or `null` when none applies. Consider using `pre_processing_function` for custom implementations or more complex combinations.              |
| statistics              | \[[Statistics Object](#bands-and-statistics)]           | Dataset statistics for the training dataset used to normalize the inputs.                                                                                                                                                                     |
| pre_processing_function | [Processing Expression](#processing-expression) \| null | Custom preprocessing function where normalization and rescaling, and any other significant operations takes place.                                                                                                                            |

Fields that accept the `null` value can be considered `null` when omitted entirely for parsing purposes.
However, setting `null` explicitly when this information is known by the model provider can help users understand
what is the expected behavior of the model. It is therefore recommended to provide `null` explicitly when applicable.

#### Bands and Statistics

Depending on the supported `stac_version` and other `stac_extensions` employed by the STAC Item using MLM,
the [STAC 1.1 - Band Object][stac-1.1-band], 
the [STAC Raster - Band Object][stac-raster-band] or
the [STAC EO - Band Object][stac-eo-band] can be used for
representing bands information, including notably the `nodata` value,
the `data_type` (see also [Data Type Enum](#data-type-enum)),
and [Common Band Names][stac-band-names].

> :information_source: <br>
> Due to how the schema for [`eo:bands`][stac-eo-band] is defined, it is not sufficient to *only* provide
> the `eo:bands` property at the STAC Item level. The schema validation of the EO extension explicitly looks
> for a corresponding set of bands under an Asset, and if none is found, it disallows `eo:bands` in the Item properties.
> Therefore, `eo:bands` should either be specified *only* under the Asset containing the `mlm:model` role
> (see [Model Asset](#model-asset)), or define them *both* under the Asset and Item properties. If the second
> approach is selected, it is recommended that the `eo:bands` under the Asset contains only the `name` or the
> `common_name` property, such that all other details about the bands are defined at the Item level.
> <br><br>
> For more details, refer to [stac-extensions/eo#12](https://github.com/stac-extensions/eo/issues/12).
> <br>
> For an example, please refer to [examples/item_eo_bands.json](examples/item_eo_bands.json).
> Notably in this example, the `assets.weights.eo:bands` property provides the `name` to fulfill the Asset requirement,
> while all additional band details are provided in `properties.eo:bands`.

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
[stac-1.1-stats]: https://github.com/radiantearth/stac-spec/blob/bands/item-spec/common-metadata.md#statistics-object
[stac-eo-band]: https://github.com/stac-extensions/eo?tab=readme-ov-file#band-object
[stac-raster-band]: https://github.com/stac-extensions/raster?tab=readme-ov-file#raster-band-object
[stac-raster-stats]: https://github.com/stac-extensions/raster?tab=readme-ov-file#statistics-object
[stac-band-names]: https://github.com/stac-extensions/eo?tab=readme-ov-file#common-band-names

#### Data Type Enum

When describing the `data_type` provided by a [Band](#bands-and-statistics), whether for defining
the [Input Structure](#input-structure-object) or the [Result Structure](#result-structure-object),
the [Data Types from the STAC Raster extension][raster-data-types] should be used if using STAC 1.0 or earlier,
and can use [Data Types from STAC 1.1 Core][stac-1.1-data-types] for later versions.
Both definitions should define equivalent values.

[raster-data-types]: https://github.com/stac-extensions/raster?tab=readme-ov-file#data-types
[stac-1.1-data-types]: https://github.com/radiantearth/stac-spec/blob/bands/item-spec/common-metadata.md#data-types

#### Input Structure Object

| Field Name | Type                                   | Description                                                                                                                                                                                                               |
|------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | \[integer]                             | **REQUIRED** Shape of the input n-dimensional array (e.g.: $B \times C \times H \times W$), including the batch size dimension. Each dimension must either be greater than 0 or -1 to indicate a variable dimension size. |
| dim_order  | \[[Dimension Order](#dimension-order)] | **REQUIRED** Order of the `shape` dimensions by name.                                                                                                                                                                     |
| data_type  | [Data Type Enum](#data-type-enum)      | **REQUIRED** The data type of values in the n-dimensional array. For model inputs, this should be the data type of the processed input supplied to the model inference function, not the data type of the source bands.   |

A common use of `-1` for one dimension of `shape` is to indicate a variable batch-size.
However, this value is not strictly reserved for the `b` dimension.
For example, if the model is capable of automatically adjusting its input layer to adapt to the provided input data,
then the corresponding dimensions that can be adapted can employ `-1` as well.

#### Dimension Order

Recommended values should use common names as much as possible to allow better interpretation by users and scripts
that could need to resolve the dimension ordering for reshaping requirements according to the ML framework employed.

Below are some notable common names recommended for use, but others can be employed as needed.

- `batch`
- `channel`
- `time`
- `height`
- `width`
- `depth`
- `token`
- `class`
- `score`
- `confidence`

For example, a tensor of multiple RBG images represented as $B \times C \times H \times W$ should 
indicate `dim_order = ["batch", "channel", "height", "width"]`.

#### Normalize Enum

Select one option from:
- `min-max`
- `z-score`
- `l1`
- `l2`
- `l2sqr`
- `hamming`
- `hamming2`
- `type-mask`
- `relative`
- `inf`
- `clip`

See [OpenCV - Normalization Flags][opencv-normalization-flags]
for details about the relevant methods. Equivalent methods from other packages are applicable as well.

When a normalization technique is specified, it is expected that the corresponding [Statistics](#bands-and-statistics)
parameters necessary to perform it would be provided for the corresponding input.
For example, the `min-max` normalization would require that at least the `minimum` and `maximum` statistic properties
are provided, while the `z-score` would require `mean` and `stddev`.

If none of the above values applies, `null` (literal, not string) can be used instead.
If a custom normalization operation, or a combination of operations (with or without [Resize](#resize-enum)),
must be defined instead, consider using a [Processing Expression](#processing-expression) reference.

[opencv-normalization-flags]: https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gad12cefbcb5291cf958a85b4b67b6149f

#### Resize Enum

Select one option from:
- `crop`
- `pad`
- `interpolation-nearest`
- `interpolation-linear`
- `interpolation-cubic`
- `interpolation-area`
- `interpolation-lanczos4`
- `interpolation-max`
- `wrap-fill-outliers`
- `wrap-inverse-map`

See [OpenCV - Interpolation Flags][opencv-interpolation-flags]
for details about the relevant methods. Equivalent methods from other packages are applicable as well.

If none of the above values applies, `null` (literal, not string) can be used instead.
If a custom rescaling operation, or a combination of operations (with or without [Normalization](#normalize-enum)),
must be defined instead, consider using a [Processing Expression](#processing-expression) reference.

[opencv-interpolation-flags]: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

#### Processing Expression

Taking inspiration from [Processing Extension - Expression Object][stac-proc-expr], the processing expression defines
at the very least a `format` and the applicable `expression` for it to perform pre/post-processing operations on MLM
inputs/outputs.

| Field Name | Type   | Description |
| ---------- | ------ | ----------- |
| format     | string | **REQUIRED** The type of the expression that is specified in the `expression` property. |
| expression | \*     | **REQUIRED** An expression compliant with the `format` specified. The expression can be any data type and depends on the `format` given, e.g. string or object. |

On top of the examples already provided by [Processing Extension - Expression Object][stac-proc-expr],
the following formats are recommended as alternative scripts and function references.

| Format   | Type   | Description                            | Expression Example                                                                                   |
|----------| ------ |----------------------------------------|------------------------------------------------------------------------------------------------------|
| `python` | string | A Python entry point reference.        | `my_package.my_module:my_processing_function` or `my_package.my_module:MyClass.my_method`            |
| `docker` | string | An URI with image and tag to a Docker. | `ghcr.io/NAMESPACE/IMAGE_NAME:latest`                                                                |
| `uri`    | string | An URI to some binary or script.       | `{"href": "https://raw.githubusercontent.com/ORG/REPO/TAG/package/cli.py", "type": "text/x-python"}` |

> :information_source: <br>
> Above definitions are only indicative, and more can be added as desired with even more custom definitions.
> It is left as an implementation detail for users to resolve how these expressions should be handled at runtime.

> :warning: <br>
> See also discussion regarding additional processing expressions:
> [stac-extensions/processing#31](https://github.com/stac-extensions/processing/issues/31)

[stac-proc-expr]: https://github.com/stac-extensions/processing#expression-object

### Model Output Object

| Field Name               | Type                                                    | Description                                                                                                                                                                     |
|--------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                     | string                                                  | **REQUIRED** Name of the output variable defined by the model. If no explicit name is defined by the model, an informative name (e.g.: `"CLASSIFICATION"`) can be used instead. |
| tasks                    | \[[Task Enum](#task-enum)]                              | **REQUIRED** Specifies the Machine Learning tasks for which the output can be used for. This can be a subset of `mlm:tasks` defined under the Item `properties` as applicable.  |
| result                   | [Result Structure Object](#result-structure-object)     | **REQUIRED** The structure that describes the resulting output arrays/tensors from one model head.                                                                              |
| description              | string                                                  | Additional details about the output such as describing its purpose or expected result that cannot be represented by other properties.                                           |
| classification:classes   | \[[Class Object](#class-object)]                        | A list of class objects adhering to the [Classification Extension](https://github.com/stac-extensions/classification).                                                          |
| post_processing_function | [Processing Expression](#processing-expression) \| null | Custom postprocessing function where normalization and rescaling, and any other significant operations takes place.                                                             |

While only `tasks` is a required field, all fields are recommended for tasks that produce a fixed
shape tensor and have output classes. Outputs that have variable dimensions, can define the `result` with the
appropriate dimension value `-1` in the `shape` field. When the model does not produce specific classes, such 
as for `regression`, `image-captioning`, `super-resolution` and some `generative` tasks, to name a few, the
`classification:classes` can be omitted.

#### Result Structure Object

| Field Name | Type                                   | Description                                                                                                                                                                                                                    |
|------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | \[integer]                             | **REQUIRED** Shape of the n-dimensional result array (e.g.: $B \times H \times W$ or $B \times C$), possibly including a batch size dimension. The dimensions must either be greater than 0 or -1 to indicate a variable size. |
| dim_order  | \[[Dimension Order](#dimension-order)] | **REQUIRED** Order of the `shape` dimensions by name for the result array.                                                                                                                                                     |
| data_type  | [Data Type Enum](#data-type-enum)      | **REQUIRED** The data type of values in the n-dimensional array. For model outputs, this should be the data type of the result of the model inference  without extra post processing.                                          |

#### Class Object

See the documentation for the
[Class Object](https://github.com/stac-extensions/classification?tab=readme-ov-file#class-object).

### Model Hyperparameters Object

The hyperparameters are an open JSON object definition that can be used to provide relevant configurations for the
model. Those can combine training details, inference runtime parameters, or both. For example, training hyperparameters
could indicate the number of epochs that were used, the optimizer employed, the number of estimators contained in an
ensemble of models, or the random state value. For inference, parameters such as the model temperature, a confidence
cut-off threshold, or a non-maximum suppression threshold to limit proposal could be specified. The specific parameter
names, and how they should be employed by the model, are specific to each implementation.

Following is an example of what the hyperparameters definition could look like:

```json
{
  "mlm:hyperparameters": {
    "nms_max_detections": 500,
    "nms_threshold": 0.25,
    "iou_threshold": 0.5,
    "random_state": 12345
  }
}
```

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
so on, as applicable for the relevant MLM Assets being described. Please refer to the following sections
for `roles` requirements by specific MLM Assets.

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

> :information_source: <br>
> (*) These roles are offered as direct conversions from the previous extension
> that provided [ML-Model Asset Roles][ml-model-asset-roles] to provide easier upgrade to the MLM extension.

[ml-model-asset-roles]: https://github.com/stac-extensions/ml-model?tab=readme-ov-file#asset-objects

### Model Asset

| Field Name        | Type                                      | Description                                                                                      |
|-------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------|
| title             | string                                    | Description of the model asset.                                                                  |
| href              | string                                    | URI to the model artifact.                                                                       |
| type              | string                                    | The media type of the artifact (see [Model Artifact Media-Type](#model-artifact-media-type).     |
| roles             | \[string]                                 | **REQUIRED** Specify `mlm:model`. Can include `["mlm:weights", "mlm:checkpoint"]` as applicable. |
| mlm:artifact_type | [Artifact Type Enum](#artifact-type-enum) | Specifies the kind of model artifact. Typically related to a particular ML framework.            |

Recommended Asset `roles` include `mlm:weights` or `mlm:checkpoint` for model weights that need to be loaded by a
model definition and `mlm:compiled` for models that can be loaded directly without an intermediate model definition.
In each case, the `mlm:model` should be applied as well to indicate that this asset represents the model.

It is also recommended to make use of the
[file](https://github.com/stac-extensions/file?tab=readme-ov-file#asset--link-object-fields)
extension for this Asset, as it can provide useful information to validate the contents of the model definition,
by comparison with fields `file:checksum` and `file:size` for example.

#### Model Artifact Media-Type

Very few ML framework, libraries or model artifacts provide explicit [IANA registered][iana-media-type] media-type
to represent the contents they handle. When those are not provided, custom media-types can be considered.
However, "*unofficial but well-established*" parameters should be reused over custom media-types when possible.

For example, the unofficial `application/octet-stream; framework=pytorch` definition is appropriate to represent a
PyTorch `.pt` file, since its underlying format is a serialized pickle structure, and its `framework` parameter
provides a clearer indication about the targeted ML framework and its contents. Since artifacts will typically be
downloaded using a request stream into a runtime environment in order to employ the model,
the `application/octet-stream` media-type is relevant for representing this type of arbitrary binary data.
Being an official media-type, it also has the benefit to increase chances that
HTTP clients will handle download of the contents appropriately when performing requests. In contrast, custom
media-types such as `application/x-pytorch` have higher chances to be considered unacceptable (HTTP 406 Not Acceptable)
by servers, which is why they should preferably be avoided.

Users can consider adding more parameters to provide additional context, such as `profile=compiled` to provide an
additional hint that the specific [PyTorch Ahead-of-Time Compilation][pytorch-aot-inductor] profile
is used for the artifact described by the media-type. However, users need to remember that those parameters are not
official. In order to validate the specific framework and artifact type employed by the model, the MLM properties
`mlm:framework` (see [MLM Fields](#item-properties-and-collection-fields)) and
`mlm:artifact_type` (see [Model Asset](#model-asset)) should be employed instead to perform this validation if needed.

[iana-media-type]: https://www.iana.org/assignments/media-types/media-types.xhtml

#### Artifact Type Enum

This value can be used to provide additional details about the specific model artifact being described.
For example, PyTorch offers [various strategies][pytorch-frameworks] for providing model definitions,
such as Pickle (`.pt`), [TorchScript][pytorch-jit-script],
or [PyTorch Ahead-of-Time Compilation][pytorch-aot-inductor] (`.pt2`) approach.
Since they all refer to the same ML framework, the [Model Artifact Media-Type](#model-artifact-media-type)
can be insufficient in this case to detect which strategy should be used with.

Following are some proposed *Artifact Type* values for corresponding approaches, but other names are
permitted as well. Note that the names are selected using the framework-specific definitions to help
the users understand the source explicitly, although this is not strictly required either.

| Artifact Type      | Description                                                                          |
|--------------------|--------------------------------------------------------------------------------------|
| `torch.save`       | A model artifact obtained by [Serialized Pickle Object][pytorch-save] (i.e.: `.pt`). |
| `torch.jit.script` | A model artifact obtained by [`TorchScript`][pytorch-jit-script].                    |
| `torch.export`     | A model artifact obtained by [`torch.export`][pytorch-export] (i.e.: `.pt2`).        |
| `torch.compile`    | A model artifact obtained by [`torch.compile`][pytorch-compile].                     |

[pytorch-compile]: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
[pytorch-export]: https://pytorch.org/docs/main/export.html
[pytorch-frameworks]: https://pytorch.org/docs/main/export.html#existing-frameworks
[pytorch-aot-inductor]: https://pytorch.org/docs/main/torch.compiler_aot_inductor.html
[pytorch-jit-script]: https://pytorch.org/docs/stable/jit.html
[pytorch-save]: https://pytorch.org/tutorials/beginner/saving_loading_models.html

### Source Code Asset

| Field Name     | Type      | Description                                                                   |
|----------------|-----------|-------------------------------------------------------------------------------|
| title          | string    | Title of the source code.                                                     |
| href           | string    | URI to the code repository, a ZIP archive, or an individual code/script file. |
| type           | string    | Media-type of the URI.                                                        |
| roles          | \[string] | **RECOMMENDED** Specify one or more of `["model", "code", "metadata"]`        |
| description    | string    | Description of the source code.                                               |
| mlm:entrypoint | string    | Specific entrypoint reference in the code to use for running model inference. |

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

| Field Name  | Type      | Description                                                                       |
|-------------|-----------|-----------------------------------------------------------------------------------|
| title       | string    | Description of the container.                                                     |
| href        | string    | URI of the published container, including the container registry, image and tag.  |
| type        | string    | Media-type of the container, typically `application/vnd.oci.image.index.v1+json`. |
| roles       | \[string] | Specify `["runtime"]` and any other custom roles.                                 |

If you're unsure how to containerize your model, we suggest starting from the latest official container image for
your framework that works with your model and pinning the container version.

Examples:
- [Pytorch Dockerhub](https://hub.docker.com/r/pytorch/pytorch/tags)
- [Pytorch Docker Run Example](https://github.com/pytorch/pytorch?tab=readme-ov-file#docker-image)
- [Tensorflow Dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/tags?page=8&ordering=last_updated)
- [Tensorflow Docker Run Example](https://www.tensorflow.org/install/docker#gpu_support)

Using a base image for a framework looks like:

```dockerfile
# In your Dockerfile, pull the latest base image with all framework dependencies including accelerator drivers
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

### Your specific environment setup to run your model
RUN pip install my_package
```

You can also use other base images. Pytorch and Tensorflow offer docker images for serving models for inference.
- [Torchserve](https://pytorch.org/serve/)
- [TFServing](https://github.com/tensorflow/serving)

## Relation Types

The following types should be used as applicable `rel` types in the
[Link Object](https://github.com/radiantearth/stac-spec/tree/master/item-spec/item-spec.md#link-object)
of STAC Items describing Band Assets that result from the inference of a model described by the MLM extension.

| Type         | Description                                              |
|--------------|----------------------------------------------------------|
| derived_from | This link points to a STAC Collection or Item using MLM. |

It is recommended that the link using `derived_from` referring to another STAC definition using the MLM extension
specifies the [`mlm:name`](#item-properties-and-collection-fields) value to make the derived reference more explicit.

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

The same checks that run as checks on PRs are part of the repository and can be run locally to verify that changes 
are valid. To run tests locally, you'll need `npm`, which is a standard part of 
any [node.js](https://nodejs.org/en/download/) installation.

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
