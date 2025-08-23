# Machine Learning Model Extension Specification

[![hackmd-github-sync-badge](https://hackmd.io/XveEXOukQ52ZdpUxT8maeA/badge)](https://hackmd.io/XveEXOukQ52ZdpUxT8maeA?both)

- **Title:** Machine Learning Model Extension
- **Identifier:** <https://stac-extensions.github.io/mlm/v1.5.0/schema.json>
- **Field Name Prefix:** mlm
- **Scope:** Collection, Item, Asset, Links
- **Extension Maturity Classification:** Candidate
- **Owner:**
  - [@fmigneault](https://github.com/fmigneault)
  - [@rbavery](https://github.com/rbavery)
  - [@ymoisan](https://github.com/ymoisan)
  - [@sfoucher](https://github.com/sfoucher)

## Contributors

<table>
    <tr>
        <td><img src="docs/static/crim.png" width="200px" alt="CRIM"></td>
        <td>
            <a href="https://www.crim.ca/en/">Computer Research Institute of Montréal</a>
            <br>
            <a href="https://www.crim.ca/fr/">Centre de Recherche Informatique de Montréal</a>
        </td>
    </tr>
    <tr>
        <td><img src="docs/static/wherobots.png" width="200px" alt="Wherobots"></td>
        <td><a href="https://wherobots.com/">Wherobots</a></td>
    </tr>
    <tr>
        <td><img src="docs/static/terradue.png" width="200px" alt="Terradue"></td>
        <td><a href="https://terradue.com/">Terradue</a></td>
    </tr>
    <tr>
        <td><img src="docs/static/nrcan.png" width="200px" alt="NRCan"></td>
        <td>
            <a href="https://natural-resources.canada.ca/">Natural Resources Canada</a>
            <br>
            <a href="https://natural-resources.canada.ca/research-centres-and-labs/canada-centre-for-mapping-and-earth-observation/25735">
                Canada Centre for Mapping and Earth Observation (CCMEO)
            </a>
        </td>
    </tr>
</table>

## Description

The STAC Machine Learning Model (MLM) Extension provides a standard set of fields to describe machine learning models
trained on overhead imagery and enable running model inference.

The main objectives of the extension are:

1) to enable building model collections that can be searched alongside associated STAC datasets
2) record all necessary bands, data variables, hyperparameters, modeling artifact locations, and
   high-level processing steps to deploy an inference service.

Specifically, this extension records the following information to make ML models searchable and reusable:

1. Sensor band specifications
2. Model input transforms including resize and normalization
3. Model output shape, data type, and its semantic interpretation
4. An optional, flexible description of the runtime environment to be able to run the model
5. Scientific references

The MLM specification is biased towards providing metadata fields for supervised machine learning models.
However, fields that relate to supervised ML are optional and users can use the fields they need for different tasks.

![STAC_MLM](./docs/static/stac_mlm.png)

<!-- lint disable -->

> Francis Charette-Migneault, Ryan Avery, Brian Pondi, Joses Omojola, Simone Vaccari, Parham Membari, Devis Peressutti, Jia Yu, and Jed Sundwall. 2024. Machine Learning Model Specification for Cataloging Spatio-Temporal Models (Demo Paper). In 3rd ACM SIGSPATIAL International Workshop on Searching and Mining Large Collections of Geospatial Data (GeoSearch’24), October 29–November 1 2024, Atlanta, GA, USA. ACM, New York, NY, USA, 4 pages. <https://doi.org/10.1145/3681769.3698586>

<!-- lint enable -->

See [Best Practices](./best-practices.md) for guidance on what other STAC extensions you should use in conjunction
with this extension as well as suggested values for specific ML framework.

The Machine Learning Model Extension purposely omits and delegates some definitions to other STAC extensions to favor
reusability and avoid metadata duplication whenever possible. A properly defined MLM STAC Item/Collection should almost
never have the Machine Learning Model Extension exclusively in `stac_extensions`.

For details about the earlier (legacy) version of the MLM Extension, formerly known as
the *Deep Learning Model Extension* (DLM), please refer to the [DLM LEGACY](./docs/legacy/dlm.md) document.
DLM was renamed to the current MLM Extension and refactored to form a cohesive definition across all machine
learning approaches, regardless of whether the approach constitutes a deep neural network or other statistical approach.
It also combines multiple definitions from the predecessor [ML-Model](https://github.com/stac-extensions/ml-model)
extension to synthesize common use cases into a single reference for "*Machine Learning Models*". For migration
details from `ml-model` to `mlm`, please refer to the [ML-Model Legacy](./docs/legacy/ml-model.md) document.

For more details about the [`stac-model`](./stac_model) Python package, which provides definitions of the MLM extension
using both [`Pydantic`](https://docs.pydantic.dev/latest/) and [`PySTAC`](https://pystac.readthedocs.io/en/stable/)
connectors, please refer to the [STAC Model](./README_STAC_MODEL.md) document.

## Resources

- **Packages**:
  - [`stac-model`](./stac_model): Model implementations using [`Pydantic`](https://docs.pydantic.dev/latest/).
    See the [STAC Model](./README_STAC_MODEL.md) document for installation and example details.
  - [`pystac.extensions.mlm`](https://github.com/stac-utils/pystac/blob/main/pystac/extensions/mlm.py): 
    Official [`pystac`](https://github.com/stac-utils/pystac) extension integration (avaiable starting
    with [`pystac v1.13.0`](https://github.com/stac-utils/pystac/blob/main/CHANGELOG.md#v1130---2025-04-15)).
- **Examples**:
  - [Local Examples](./examples): Demonstrates different combinations of STAC extensions used along MLM.
  - [Item examples](https://huggingface.co/wherobots/mlm-stac): Demonstrates `scene-classification`,
      `object-detection`, and `semantic-segmentation` tasks with real world use of the
      MLM extension for describing models run on
      [WherobotsAI Raster Inference](https://wherobots.com/wherobotsai-for-raster-inference/).
  - [Collection example](examples/collection.json): Shows the basic usage of the extension in a STAC Collection.
- **Documentation**:
  - [JSON Schema](https://stac-extensions.github.io/mlm/)
  - [Changelog](./CHANGELOG.md)
  - [Best Practices](./best-practices.md): Recommended combinations of MLM with
    other [STAC Extensions](https://stac-extensions.github.io/) for efficient integration into the STAC ecosystem.
  - [Open access paper](https://dl.acm.org/doi/10.1145/3681769.3698586) describing
    version [`1.3.0`](https://github.com/stac-extensions/mlm/blob/main/CHANGELOG.md#v130) of the extension.
  - [SigSpatial 2024 GeoSearch Workshop presentation](/docs/static/sigspatial_2024_mlm.pdf)
- **Tools**:
  - [MLM Form Filler](https://mlm-form.vercel.app/) a two-page app to fill out and validate MLM STAC Item metadata. <br>
    Check out the [wherobots/mlm-form](https://github.com/wherobots/mlm-form) repository if you have questions, issues,
    or want to contribute.

## Item Properties and Collection Fields

The fields in the table below can be used in these parts of STAC documents:

- [ ] Catalogs
- [x] Collections
- [x] Item Properties (incl. Summaries in Collections)
- [x] Assets (for both Collections and Items, incl. [Item-Assets][item-assets] definitions in Collections)
- [ ] Links

[item-assets]: https://github.com/stac-extensions/item-assets

| Field Name                                | Type                                                          | Description                                                                                                                                                                                                                                                                                 |
|-------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlm:name <sup>[\[1\]][1]</sup>            | string                                                        | **REQUIRED** A name for the model. This can include, but must be distinct, from simply naming the model architecture. If there is a publication or other published work related to the model, use the official name of the model.                                                           |
| mlm:architecture                          | [Model Architecture](#model-architecture) string              | **REQUIRED** A generic and well established architecture name of the model.                                                                                                                                                                                                                 |
| mlm:tasks                                 | \[[Task Enum](#task-enum)]                                    | **REQUIRED** Specifies the Machine Learning tasks for which the model can be used for. If multi-tasks outputs are provided by distinct model heads, specify all available tasks under the main properties and specify respective tasks in each [Model Output Object](#model-output-object). |
| mlm:framework                             | string                                                        | Framework used to train the model (ex: PyTorch, TensorFlow). Typically, this will align with the applied `mlm:artifact_type` of the [Model Asset](#model-asset).                                                                                                                            |
| mlm:framework_version                     | string                                                        | The `framework` library version. Some models require a specific version of the machine learning `framework` to run.                                                                                                                                                                         |
| mlm:memory_size                           | integer                                                       | The in-memory size of the model on the accelerator during inference (bytes).                                                                                                                                                                                                                |
| mlm:total_parameters                      | integer                                                       | Total number of model parameters, including trainable and non-trainable parameters.                                                                                                                                                                                                         |
| mlm:pretrained                            | boolean                                                       | Indicates if the model was pretrained. If the model was pretrained, consider providing `pretrained_source` if it is known.                                                                                                                                                                  |
| mlm:pretrained_source                     | string \| null                                                | The source of the pretraining. Can refer to popular pretraining datasets by name (i.e. Imagenet) or less known datasets by URL and description. If trained from scratch (i.e.: `pretrained = false`), the `null` value should be set explicitly.                                            |
| mlm:batch_size_suggestion                 | integer                                                       | A suggested batch size for the accelerator and summarized hardware.                                                                                                                                                                                                                         |
| mlm:accelerator                           | [Accelerator Type Enum](#accelerator-type-enum) \| null       | The intended computational hardware that runs inference. If undefined or set to `null` explicitly, the model does not require any specific accelerator.                                                                                                                                     |
| mlm:accelerator_constrained               | boolean                                                       | Indicates if the intended `accelerator` is the only `accelerator` that can run inference. If undefined, it should be assumed `false`.                                                                                                                                                       |
| mlm:accelerator_summary                   | string                                                        | A high level description of the `accelerator`, such as its specific generation, or other relevant inference details.                                                                                                                                                                        |
| mlm:accelerator_count                     | integer                                                       | A minimum amount of `accelerator` instances required to run the model.                                                                                                                                                                                                                      |
| mlm:input <sup>[\[1\]][1]</sup>           | \[[Model Input Object](#model-input-object)]                  | **REQUIRED** Describes the transformation between the EO data and the model input.                                                                                                                                                                                                          |
| mlm:output <sup>[\[1\]][1]</sup>          | \[[Model Output Object](#model-output-object)]                | **REQUIRED** Describes each model output and how to interpret it.                                                                                                                                                                                                                           |
| mlm:hyperparameters <sup>[\[1\]][1]</sup> | [Model Hyperparameters Object](#model-hyperparameters-object) | Additional hyperparameters relevant for the model.                                                                                                                                                                                                                                          |

<!-- special heading is done on purpose to correctly render and redirect on GitHub while avoiding linting issues -->

[1]: #notes

### Notes

<b><sup>[1][1]</sup> Fields allowed only in Item `properties`</b>

<!-- lint disable no-undefined-references -->

> [!NOTE]
> Unless stated otherwise by <sup>[\[1\]][1]</sup> in the table, fields can be used at either the Item or Asset level.
> <br><br>
> To decide whether above fields should be applied under Item `properties` or under respective Assets, the context of
> each field must be considered. For example, the `mlm:name` should always be provided in the Item `properties`, since
> it relates to the model as a whole. In contrast, some models could support multiple `mlm:accelerator`, which could be
> handled by distinct source code represented by different Assets. In such case, `mlm:accelerator` definitions should be
> nested under their relevant Asset. If a field is defined both at the Item and Asset level, the value at the Asset
> level would be considered for that specific Asset, and the value at the Item level would be used for other Assets that
> did not override it for their respective reference. For some of the fields, further details are provided in following
> sections to provide more precisions regarding some potentially ambiguous use cases.

<!-- lint enable no-undefined-references -->

In addition, fields from the multiple relevant extensions should be defined as applicable. See
[Best Practices - Recommended Extensions to Compose with the ML Model Extension](best-practices.md#recommended-extensions-to-compose-with-the-mlm-extension)
for more details.

For the [Extent Object][stac-extent]
in STAC Collections and the corresponding spatial and temporal fields in Items, please refer to section
[Best Practices - Using STAC Common Metadata Fields for the ML Model Extension][stac-mlm-meta].

[stac-mlm-meta]: best-practices.md#using-stac-common-metadata-fields-for-the-mlm-extension

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
| `downscaling`           | *n/a*                       | Specific task reduces the coarser data variables at larger scale to a smaller and finer scale of higher resolution.      |

If the task falls within the category of supervised machine learning and uses labels during training,
this should align with the `label:tasks` values defined in [STAC Label Extension][stac-ext-label-props] for relevant
STAC Collections and Items published with the model described by this extension.

It is to be noted that multiple "*generic*" tasks names (`classification`, `detection`, etc.) are defined to allow
correspondence with `label:tasks`, but these can lead to some ambiguity depending on context. For example, a model
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

- [`PyTorch`](https://github.com/pytorch/pytorch)
- [`TensorFlow`](https://github.com/tensorflow/tensorflow)
- [`scikit-learn`](https://github.com/scikit-learn/scikit-learn)
- [`Hugging Face`](https://github.com/huggingface/)
- [`Keras`](https://github.com/keras-team/keras)
- [`ONNX`](https://github.com/onnx/onnx)
- [`rgee`](https://github.com/r-spatial/rgee)
- [`spatialRF`](https://github.com/BlasBenito/spatialRF)
- [`JAX`](https://github.com/jax-ml/jax)
- [`Flax`](https://github.com/google/flax)
- [`MXNet`](https://github.com/apache/mxnet)
- [`Caffe`](https://github.com/BVLC/caffe)
- [`PyMC`](https://github.com/pymc-devs/pymc)
- [`Weka`](https://ml.cms.waikato.ac.nz/weka/)
- [`Paddle`](https://github.com/PaddlePaddle/Paddle)

### Accelerator Type Enum

It is recommended to define `accelerator` with one of the following values:

- `amd64` models compatible with AMD or Intel CPUs (no hardware specific optimizations)
- `cpu` an alias for the `amd64` accelerator type
- `cuda` models compatible with NVIDIA GPUs
- `xla` models compiled with XLA. Models trained on TPUs are typically compiled with XLA.
- `amd-rocm` models trained on AMD GPUs
- `intel-ipex-cpu` for models optimized with IPEX for Intel CPUs
- `intel-ipex-gpu` for models optimized with IPEX for Intel GPUs
- `macos-arm` for models trained on Apple Silicon

<!-- lint disable no-undefined-references -->

> [!WARNING]
> If `mlm:accelerator = amd64`, this explicitly indicates that the model does not (and will not try to) use any
> accelerator, even if some are available from the runtime environment. This is to be distinguished from
> the value `mlm:accelerator = null`, which means that the model *could* make use of some accelerators if provided,
> but is not constrained by any specific one. To improve comprehension by users, it is recommended that any model
> using `mlm:accelerator = amd64` also set explicitly `mlm:accelerator_constrained = true` to illustrate that the
> model **WILL NOT** use accelerators, although the hardware resolution should be identical nonetheless.

<!-- lint enable no-undefined-references -->

When `mlm:accelerator = null` is employed, the value of `mlm:accelerator_constrained` can be ignored, since even if
set to `true`, there would be no `accelerator` to contain against. To avoid confusion, it is suggested to set the
`mlm:accelerator_constrained = false` or omit the field entirely in this case.

### Model Input Object

| Field Name              | Type                                                                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                    | string                                                                                                        | **REQUIRED** Name of the input variable defined by the model. If no explicit name is defined by the model, an informative name (e.g.: `"RGB Time Series"`) can be used instead.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| description             | string                                                                                                        | Additional details about the input such as describing its purpose or expected source that cannot be represented by other properties.                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| bands                   | \[string \| [Model Band Object](#model-band-or-variable-object)]                                              | If applicable, the raster band references used to train, fine-tune or perform inference with the model, which may be all or a subset of [Band Object](#bands-and-statistics). If none applies for the input, use an empty array or omit the field entirely. <sup>[\[1\]][note-band-vars]</sup>                                                                                                                                                                                                                                                                                                            |
| variables               | \[string \| [Model Variable Object](#model-band-or-variable-object)]                                          | If applicable, the data variable references used to train, fine-tune or perform inference with the model, which may be all or a subset of [Data Variables](#data-variables). If none applies for the input, use an empty array or omit the field entirely. <sup>[\[1\]][note-band-vars]</sup>                                                                                                                                                                                                                                                                                                             |
| input                   | [Input Structure Object](#input-structure-object)                                                             | **REQUIRED** The N-dimensional array definition that describes the shape, dimension ordering, and data type.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| value_scaling           | \[[Value Scaling Object](#value-scaling-object)] \| null                                                      | Method to scale, normalize, or standardize the data inputs values, across dimensions, per corresponding dimension index, or `null` if none applies. These values often correspond to dataset or sensor statistics employed for training the model, but can come from another source as needed by the model definition. Consider using `pre_processing_function` for custom implementations or more complex combinations.                                                                                                                                                                                  |
| resize_type             | [Resize Enum](#resize-enum) \| null                                                                           | High-level descriptor of the resize method to modify the input dimensions shape. Select an appropriate option or `null` when none applies. Consider using `pre_processing_function` for custom implementations or more complex combinations.                                                                                                                                                                                                                                                                                                                                                              |
| pre_processing_function | [Processing Expression](#processing-expression) \| \[[Processing Expression](#processing-expression)] \| null | Custom preprocessing function(s) where rescaling, resize, and any other significant data preparation operations takes place. The `pre_processing_function` should be applied over all available `bands` and/or `variables` if any are defined. For operations applied only to specific `bands` or `variables`, employ the `expression` of [Model Band or Variable Object](#model-band-or-variable-object) instead. If multiple expressions are provided using the array structure, they should be applied sequentially in the same order, with each transformation result feeding into the following one. |

<!-- lint disable no-undefined-references -->

> [!TIP]
> Fields that accept the `null` value can be considered `null` when omitted entirely for parsing purposes.
> However, setting `null` explicitly when this information is known by the model provider can help users understand
> what is the expected behavior of the model. It is therefore recommended to provide `null` explicitly when applicable.

<!-- lint enable no-undefined-references -->

[note-band-vars]: #bands-and-variables

#### Bands and Variables

<!-- lint disable no-undefined-references -->

> [!IMPORTANT]
> Either `bands`, `variables`, both of them, or none at all can be specified.
> Whichever combination of fields is applied depends on the model definition and what best describes the data
> that the model interacts with to ensure the best clarity and comprehension by users.
> 
> If both `bands` and `variables` are supplied simultaneously, distinct names should be employed
> to allow distinguishing between them and the references of corresponding STAC extensions they refer to
> (see [Band Object](#bands-and-statistics) and [Data Variable](#data-variables)).
> 
> Although `bands` and `variables` are not required, supplying them is **STRONGLY RECOMMENDED** if they are applicable.
> This is to ensure that the users can properly understand which format and data structures are expected or produced by
> the model.
> 
> When either `bands` or `variables` are provided (non-empty arrays), they **MUST** refer to corresponding
> [Band Object](#bands-and-statistics) and [Data Variable](#data-variables) respectively. The MLM extension will
> explicitly validate that corresponding STAC extensions to describe them are present in the STAC Item for consistency.
> Furthermore, the corresponding dimension names (indicated in [Dimension Order](#dimension-order)) that characterize
> the [Input Structure](#input-structure-object) or the [Result Structure](#result-structure-object) **MUST** include
> `"bands"` and/or `"variables"` when they are provided and defined for a given input or output.
> 
> If no `bands` or `variables` are needed to describe dimensions (e.g.: describing generic text or floating value),
> it is *RECOMMENDED* to set both of them to an empty array (`[]`) to be explicit about it. However, omitting the fields
> entirely is also permitted. If omitted, `bands` and `variables` properties should be interpreted as if empty arrays
> were explicitly indicated.

> [!IMPORTANT]
> The order of elements within `bands` and `variables` fields is important,
> notably when the [Input Structure Object](#input-structure-object) or the [Result Structure](#result-structure-object)
> references `"bands"` and/or `"variables"`
> as dimensions of stacked bands/variables within its [Dimension Order](#dimension-order) property.

> [!WARNING]
> Due to above requirements regarding `"bands"` and `"variables"` dimension names and their required cross-references
> to valid definitions under the corresponding properties, these names are reserved and cannot be used as generic
> dimension names while omitting their definitions.

> [!NOTE]
> For convenience, each item in `bands` and `variables` can be defined directly as a single `string` value or using an
> explicit [Model Band or Variable Object](#model-band-or-variable-object) with a `name` property. In each case, the
> implicit or explicit `name` should map to a corresponding [Band Object](#bands-and-statistics)
> or [Data Variable](#data-variables) as applicable, with further metadata describing them.

<!-- lint enable no-undefined-references -->

#### Bands and Statistics

Depending on the supported `stac_version` and other `stac_extensions` employed by the STAC Item using MLM,
the [STAC 1.1 - Band Object][stac-1.1-band],
the [STAC Raster - Band Object][stac-raster-band] or
the [STAC EO - Band Object][stac-eo-band] can be used for
representing bands information, including notably the `nodata` value,
the `data_type` (see also [Data Type Enum](#data-type-enum)),
and [Common Band Names][stac-band-names].

<!-- lint disable no-undefined-references -->

> [!WARNING]
> Only versions `v1.x` of `eo` and `raster` are supported to provide `mlm:input` band references.
> Versions `2.x` of those extensions rely on the [STAC 1.1 - Band Object][stac-1.1-band] instead.
> If those versions are desired, consider migrating your MLM definition to use [STAC 1.1 - Band Object][stac-1.1-band]
> as well for referencing `mlm:input` with band names.

> [!NOTE]
> Due to how the schema for [`eo:bands`][stac-eo-band] is defined, it is not sufficient to *only* provide
> the `eo:bands` property at the STAC Item level. The schema validation of the EO extension explicitly looks
> for a corresponding set of bands under an Asset, and if none is found, it disallows `eo:bands` in the Item properties.
> Therefore, `eo:bands` should either be specified *only* under the Asset containing the `mlm:model` role
> (see [Model Asset](#model-asset)), or define them *both* under the Asset and Item properties. If the second
> approach is selected, it is recommended that the `eo:bands` under the Asset contains only the `name` or the
> `common_name` property, such that all other details about the bands are defined and cross-referenced by name
> with the Item-level band definitions. An example of such representation is provided in
> [examples/item_eo_bands_summarized.json](examples/item_eo_bands_summarized.json).
> For an example where `eo:bands` are entirely defined in the Asset on their own, please refer to
> [examples/item_eo_bands.json](examples/item_eo_bands.json) instead.
> <br><br>
> For more details, refer to [stac-extensions/eo#12](https://github.com/stac-extensions/eo/issues/12).
> <br>

> [!NOTE]
> When using `raster:bands`, and additional `name` parameter **MUST** be provided for each band. This parameter
> is not defined in `raster` extension itself, but is permitted. This addition is required to ensure
> that `mlm:input` bands referenced by name can be associated to their respective `raster:bands` definitions.

<!-- lint enable no-undefined-references -->

Only bands used as input to the model should be included in the MLM `bands` field.
To avoid duplicating the information, MLM only uses the `name` of whichever "Band Object" is defined in the STAC Item.
An input's `bands` definition can either be a plain `string` or a [Model Band Object](#model-band-or-variable-object).
When a `string` is employed directly, the value should be implicitly mapped to the `name` property of the
explicit object representation.

One distinction from the [STAC 1.1 - Band Object][stac-1.1-band] in MLM is that [Band Statistics][stac-1.1-stats] object
(or the corresponding [STAC Raster - Statistics][stac-raster-stats] for STAC 1.0) are not
defined at the "Band Object" level, but at the [Model Input](#model-input-object) level.
This is because, in machine learning, it is common to need overall statistics for the dataset used to train the model
to normalize all bands, rather than normalizing the values over a single product. Furthermore, statistics could be
applied differently for distinct [Model Input](#model-input-object) definitions, in order to adjust for intrinsic
properties of the model.

Another distinction is that, depending on the model, statistics could apply to some inputs that have no reference to
any `bands` definition. In such case, defining statistics under `bands` would not be possible, or would intrude
ambiguous definitions.

Finally, contrary to the "`statistics`" property name employed by [Band Statistics][stac-1.1-stats], MLM employs the
distinct property name `value_scaling`, although similar `minimum`, `maximum`, etc. sub-fields are employed.
This is done explicitly to disambiguate "informative" band statistics from "applied scaling operations" required
by the model inputs. This highlights the fact that `value_scaling` are not *necessarily* equal
to [Band Statistics][stac-1.1-stats] values, although they are often equal in practice due to the applicable
value-range domains they represent. Also, this allows addressing special scaling cases, using additional properties
unavailable from [Band Statistics][stac-1.1-stats], such as `value`-specific scaling
(see [Value Scaling Object](#value-scaling-object) for more details).

[stac-1.1-band]: https://github.com/radiantearth/stac-spec/blob/v1.1.0/commons/common-metadata.md#bands

[stac-1.1-stats]: https://github.com/radiantearth/stac-spec/blob/v1.1.0/commons/common-metadata.md#statistics-object

[stac-eo-band]: https://github.com/stac-extensions/eo/tree/v1.1.0#band-object

[stac-raster-band]: https://github.com/stac-extensions/raster/tree/v1.1.0#raster-band-object

[stac-raster-stats]: https://github.com/stac-extensions/raster/tree/v1.1.0#statistics-object

[stac-band-names]: https://github.com/stac-extensions/eo#common-band-names

#### Data Variables

Defining a `variables` array in [Model Input Object](#model-input-object) allows to reference extended data variables
definitions that better describe the structure of the data the model expects as input. Similarly, `variables` applied
in [Model Output Object](#model-output-object) describes specific variables produced by a model output.

The variable `name` referenced in the [Model Variable Object](#model-band-or-variable-object) should directly
reference to a corresponding key of the [`cube:variables`][datacube-fields] mapping.
The [Variable Object][datacube-variable-object] under each [`cube:variables`][datacube-fields] key should conform
to the STAC [datacube][datacube] extension.

<!-- lint disable no-undefined-references -->

> [!NOTE]
> Because each [Variable Object][datacube-variable-object] **requires** a ``dimensions`` property, the corresponding
> [`cube:dimensions`][datacube-fields] mapping should also be provided in the STAC Item definition containing the
> MLM metadata.

<!-- lint enable no-undefined-references -->

An example of `variables` applied to a model definition
is provided in [examples/item_datacube_variables.json](examples/item_datacube_variables.json).

[datacube]: https://github.com/stac-extensions/datacube
[datacube-fields]: https://github.com/stac-extensions/datacube?tab=readme-ov-file#fields
[datacube-variable-object]: https://github.com/stac-extensions/datacube?tab=readme-ov-file#variable-object

#### Model Band or Variable Object

| Field Name | Type   | Description                                                                                                                                                      |
|------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name       | string | **REQUIRED** Name of the band or variable referring to an extended definition (see [Bands](#bands-and-statistics) and [Data Variable](#data-variables) details). |
| format     | string | The type of expression that is specified in the `expression` property.                                                                                           |
| expression | \*     | An expression compliant with the `format` specified. The expression can be applied to any data type and depends on the `format` given.                           |

<!-- lint disable no-undefined-references -->

> [!NOTE]
> Although `format` and `expression` are not required in this context, they are mutually dependent on each other. <br>
> See also [Processing Expression](#processing-expression) for more details and examples.

<!-- lint enable no-undefined-references -->

The `format` and `expression` properties can serve multiple purpose.

1. Applying a pre-processing step to a specific *input* band or variable,
   in contrast to [`pre_processing_function`](#model-input-object) applied over all bands and variables.
   For example, reshaping a band to align its dimensions with other bands before stacking them,
   or realigning a data variable over a common spatio-temporal grid. 

2. Applying a post-processing step to a specific *output* band or variable,
   in contrast to [`post_processing_function`](#model-output-object) applied over all bands and variables.
   For example, applying an inverse normalization operation to rescale outputs to physical properties.

3. Defining a derived-data operation or manipulation that produces a virtual band or prepares variable references.
   For example, computing an index that applies an arithmetic combination of multiple bands (e.g.: NDVI), or decoupling
   a specific sub-axis of an auxiliary data variable.

4. Referring to a custom script or function that performs more complex data preparation 
  or that requires a dedicated runtime environment and software dependencies.

<!-- lint disable no-undefined-references -->

> [!TIP]
> For examples using expressions referring to custom Python scripts, Docker images or file URIs,
> see [stac-extensions/processing#31](https://github.com/stac-extensions/processing/issues/31)
> and [stac-extensions/mlm#28](https://github.com/stac-extensions/mlm/issues/28).
> 
> For concrete examples in the context of MLM, see:
> - [examples/item_bands_expression.json](examples/item_bands_expression.json)
> - [examples/item_datacube_variables.json](examples/item_datacube_variables.json)

<!-- lint enable no-undefined-references -->

#### Data Type Enum

When describing the `data_type` provided by a [Band](#bands-and-statistics) or [Variable](#data-variables), 
whether for defining the [Input Structure](#input-structure-object) or the [Result Structure](#result-structure-object),
the [Data Types from the STAC Raster extension][raster-data-types] 
and [Data Types from the STAC DataCube Extension][datacube-data-types]
should be used respectively if using STAC 1.0 or earlier,
and can use [Data Types from STAC 1.1 Core][stac-1.1-data-types] for later versions.
All definitions should define equivalent and interchangeable `data_type` values.

[raster-data-types]: https://github.com/stac-extensions/raster?tab=readme-ov-file#data-types

[datacube-data-types]: https://github.com/stac-extensions/datacube?tab=readme-ov-file#variable-object

[stac-1.1-data-types]: https://github.com/radiantearth/stac-spec/blob/bands/item-spec/common-metadata.md#data-types

#### Input Structure Object

| Field Name | Type                                   | Description                                                                                                                                                                                                                 |
|------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | \[integer]                             | **REQUIRED** Shape of the input n-dimensional array (e.g.: $`B \times C \times H \times W`$), including the batch size dimension. Each dimension must either be greater than 0 or -1 to indicate a variable dimension size. |
| dim_order  | \[[Dimension Order](#dimension-order)] | **REQUIRED** Order of the `shape` dimensions by name.                                                                                                                                                                       |
| data_type  | [Data Type Enum](#data-type-enum)      | **REQUIRED** The data type of values in the n-dimensional array. For model inputs, this should be the data type of the processed input supplied to the model inference function, not the data type of the source bands.     |

A common use of `-1` for one dimension of `shape` is to indicate a variable batch-size.
However, this value is not strictly reserved for the `b` dimension.
For example, if the model is capable of automatically adjusting its input layer to adapt to the provided input data,
then the corresponding dimensions that can be adapted can employ `-1` as well.

#### Dimension Order

Recommended values should use common names as much as possible to allow better interpretation by users and scripts
that could need to resolve the dimension ordering for reshaping requirements according to the ML framework employed.

Below are some notable common names recommended for use, but others can be employed as needed.

- `batch` (`B`)
- `channel` (`C`)
- `bands` (referring to multiple [Band Object](#model-band-or-variable-object) definitions)
- `variables` (referring to multiple [Variable Object](#model-band-or-variable-object) definitions)
- `temperature` (`T`)
- `pressure` (`P`)
- `time` (`t`)
- `latitude` (`lat`)
- `longitude` (`lon`)
- `altitude` (`alt`)
- `height` (`h`)
- `width` (`w`)
- `depth` (`D`)
- `x`, `y`, `z` (2D or 3D spatial dimensions)
- `token`
- `class`
- `score`
- `confidence`
- `embedding`

For example, a tensor of multiple RBG images represented as $`B \times C \times H \times W`$ should
indicate `dim_order = ["batch", "channel", "height", "width"]`.

<!-- lint disable no-undefined-references -->

> [!NOTE]
> When selecting or interpreting dimension names, users should take careful consideration over the multiple
> interpretations they might introduce based on context. For example, `h`/`height` could either represent the height of
> the image (often mapped to `y` dimension in geomatics) or it could represent the literal height/altitude of a measure.

<!-- lint enable no-undefined-references -->

#### Value Scaling Object

Select one option from:

| `type`       | Required Properties                             | Scaling Operation                                            |
|--------------|-------------------------------------------------|--------------------------------------------------------------|
| `min-max`    | `minimum`, `maximum`                            | $`(data - minimum) / (maximum - minimum)`$                   |
| `z-score`    | `mean`, `stddev`                                | $`(data - mean) / stddev`$                                   |
| `clip`       | `minimum`, `maximum`                            | $`\min(\max(data, minimum), maximum)`$                       |
| `clip-min`   | `minimum`                                       | $`\max(data, minimum)`$                                      |
| `clip-max`   | `maximum`                                       | $`\min(data, maximum)`$                                      |
| `offset`     | `value`                                         | $`data - value`$                                             |
| `scale`      | `value`                                         | $`data / value`$                                             |
| `processing` | [Processing Expression](#processing-expression) | *according to the interpretation of* `processing:expression` |

When a scaling `type` approach is specified, it is expected that the parameters necessary
to perform their calculation are provided for the corresponding input dimension data.

If none of the above values applies for a given dimension, `type: null` (literal `null`, not string) should be
used instead. If none of the input dimension require scaling, the entire definition can be defined
as `value_scaling: null` or be omitted entirely.

When `value_scaling` is specified, the amount of objects defined in the array must match the size of
the bands/channels/dimensions described by the [Model Input](#model-input-object). However, the `value_scaling` array
is allowed to contain a single object if the entire input must be rescaled using the same definition across all
dimensions. In such case, implicit broadcasting of the unique [Value Scaling Object](#value-scaling-object) should be
performed for all applicable dimensions when running inference with the model.

If a custom scaling operation, or a combination of more complex operations (with or without [Resize](#resize-enum)),
must be defined instead, a [Processing Expression](#processing-expression) reference can be specified in place of
the [Value Scaling Object](#value-scaling-object) of the respective input dimension, as shown below.

```json
{
  "value_scaling": [
    {"type": "min-max", "minimum": 0, "maximum": 255},
    {"type": "clip", "minimum": 0, "maximum": 255},
    {"type": "processing", "format": "gdal-calc", "expression": "A * logical_or( A<=177, A>=185 )"}
  ]
}
```

For operations such as L1 or L2 normalization, [Processing Expression](#processing-expression) should also be employed.
This is because, depending on the [Model Input](#model-input-object) dimensions and reference data, there is an
ambiguity regarding "how" and "where" such normalization functions must be applied against the input data.
A custom mathematical expression should provide explicitly the data manipulation and normalization strategy.

In situations of very complex `value_scaling` operations, which cannot be represented by any of the previous definition,
a `pre_processing_function` should be used instead (see [Model Input Object](#model-input-object)).

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
If a custom rescaling operation, or a combination of operations
(with or without [Value Scaling](#value-scaling-object)),
must be defined instead, consider using a [Processing Expression](#processing-expression) reference.

[opencv-interpolation-flags]: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

#### Processing Expression

Taking inspiration from [Processing Extension - Expression Object][stac-proc-expr], the processing expression defines
at the very least a `format` and the applicable `expression` for it to perform pre/post-processing operations on MLM
inputs/outputs.

| Field Name | Type   | Description                                                                                                                                                     |
|------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| format     | string | **REQUIRED** The type of the expression that is specified in the `expression` property.                                                                         |
| expression | \*     | **REQUIRED** An expression compliant with the `format` specified. The expression can be any data type and depends on the `format` given, e.g. string or object. |

On top of the examples already provided by [Processing Extension - Expression Object][stac-proc-expr],
the following formats are recommended as alternative scripts and function references.

| Format   | Type   | Description                            | Expression Example                                                                                   |
|----------|--------|----------------------------------------|------------------------------------------------------------------------------------------------------|
| `python` | string | A Python entry point reference.        | `my_package.my_module:my_processing_function` or `my_package.my_module:MyClass.my_method`            |
| `docker` | string | An URI with image and tag to a Docker. | `ghcr.io/NAMESPACE/IMAGE_NAME:latest`                                                                |
| `uri`    | string | An URI to some binary or script.       | `{"href": "https://raw.githubusercontent.com/ORG/REPO/TAG/package/cli.py", "type": "text/x-python"}` |

<!-- lint disable no-undefined-references -->

> [!NOTE]
> Above definitions are only indicative, and more can be added as desired with even more custom definitions.
> It is left as an implementation detail for users to resolve how these expressions should be handled at runtime.

> [!WARNING]
> See also discussion regarding additional processing expressions:
> [stac-extensions/processing#31](https://github.com/stac-extensions/processing/issues/31)

<!-- lint enable no-undefined-references -->

[stac-proc-expr]: https://github.com/stac-extensions/processing#expression-object

### Model Output Object

| Field Name               | Type                                                                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name                     | string                                                                                                        | **REQUIRED** Name of the output variable defined by the model. If no explicit name is defined by the model, an informative name (e.g.: `"CLASSIFICATION"`) can be used instead.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| description              | string                                                                                                        | Additional details about the output such as describing its purpose or expected result that cannot be represented by other properties.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| tasks                    | \[[Task Enum](#task-enum)]                                                                                    | **REQUIRED** Specifies the Machine Learning tasks for which the output can be used for. This can be a subset of `mlm:tasks` defined under the Item `properties` as applicable.                                                                                                                                                                                                                                                                                                                                                                                                                   |
| result                   | [Result Structure Object](#result-structure-object)                                                           | **REQUIRED** The structure that describes the resulting output arrays/tensors from one model head.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| bands                    | \[string \| [Model Band Object](#model-band-or-variable-object)]                                              | If applicable, raster band references produced by the model, which may be all or a subset of [Band Object](#bands-and-statistics). Can be omitted or set explicitly to an empty array if none applies. <sup>[\[1\]][note-band-vars]</sup>                                                                                                                                                                                                                                                                                                                                                        |
| variables                | \[string \| [Model Variable Object](#model-band-or-variable-object)]                                          | If applicable, data variable references produced by the model, which may be all or a subset of [Data Variables](#data-variables). Can be omitted or set explicitly to an empty array if none applies. <sup>[\[1\]][note-band-vars]</sup>                                                                                                                                                                                                                                                                                                                                                         |
| classification:classes   | \[[Class Object](#class-object)]                                                                              | A list of class objects adhering to the [Classification Extension](https://github.com/stac-extensions/classification).                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| post_processing_function | [Processing Expression](#processing-expression) \| \[[Processing Expression](#processing-expression)] \| null | Custom postprocessing function(s) where normalization, rescaling, or any other significant operations takes place. The `post_processing_function` should be applied over all available `bands` and/or `variables` if any are defined. For operations applied only to specific `bands` or `variables`, employ the `expression` of [Model Band or Variable Object](#model-band-or-variable-object) instead. If multiple expressions are provided using the array structure, they should be applied sequentially in the same order, with each transformation result feeding into the following one. |

While only `tasks` and `result` are required, all fields are recommended for tasks that produce a fixed
shape tensor and have output classes. Outputs that have variable dimension sizes can define the `result` with the
appropriate dimension value `-1` in the `shape` field, such as when a model automatically handles predictions over 
of multiple samples in variable batch size. If the batch is fixed, the `shape` should include that value explicitly.

When the model does not produce specific classes, such as for `regression`, `image-captioning`, `super-resolution`
and some `generative` tasks, to name a few, the `classification:classes` can be omitted.

<!-- lint disable no-undefined-references -->

> [!IMPORTANT]
> The order of elements within `bands` and `variables` fields is important, 
> notably when the [Result Structure Object](#result-structure-object) references `"bands"` and/or `"variables"`
> as dimensions of stacked bands/variables within its [Dimension Order](#dimension-order) property.

<!-- lint enable no-undefined-references -->

#### Result Structure Object

| Field Name | Type                                   | Description                                                                                                                                                                                                                        |
|------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| shape      | \[integer]                             | **REQUIRED** Shape of the n-dimensional result array (e.g.: $`B \times H \times W`$ or $`B \times C`$), possibly including a batch size dimension. The dimensions must either be greater than 0 or -1 to indicate a variable size. |
| dim_order  | \[[Dimension Order](#dimension-order)] | **REQUIRED** Order of the `shape` dimensions by name for the result array.                                                                                                                                                         |
| data_type  | [Data Type Enum](#data-type-enum)      | **REQUIRED** The data type of values in the n-dimensional array. For model outputs, this should be the data type of the result of the model inference  without extra post processing.                                              |

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

A valid STAC MLM Item definition requires at least one Asset with the `mlm:model` role, as well as,
an accompanying `mlm:artifact_type` property that describes how to employ it.
An Asset described with this role is considered the "*main*" [Model Asset](#model-asset) being described by
the STAC Item definition. Typically, there will be only one asset containing the `mlm:model` role.
However, multiple Assets employing the `mlm:model` role are permitted to provide alternate interfaces of the same model
(e.g.: using different frameworks or compilations), but those assets should have exactly the same model interfaces
(i.e.: identical `mlm:input`, `mlm:output`, etc.). In such case, the `mlm:artifact_type` property should be used to
distinguish them.

Additional definitions such as the [Source Code Asset](#source-code-asset) and the [Container Asset](#container-asset)
are considered "*side-car*" Assets that help understand how to employ the model, such as through the reference training
script that produced the model or a preconfigured inference runtime environment. These additional Assets are optional,
but it is *STRONGLY RECOMMENDED* to provide them in order to help correct adoption and use of the described model
by users.

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

| Asset Role                 | Additional Roles        | Description                                                                              |
|----------------------------|-------------------------|------------------------------------------------------------------------------------------|
| mlm:inference-runtime (\*) | `runtime`               | Describes an Asset that provides runtime reference to perform model inference.           |
| mlm:training-runtime (\*)  | `runtime`               | Describes an Asset that provides runtime reference to perform model training.            |
| mlm:checkpoint (\*)        | `weights`, `checkpoint` | Describes an Asset that provides a model checkpoint with embedded model configurations.  |
| mlm:weights                | `weights`, `checkpoint` | Describes an Asset that provides a model weights (typically some Tensor representation). |
| mlm:model                  | `model`                 | **REQUIRED** Role for [Model Asset](#model-asset).                                       |
| mlm:source_code            | `code`                  | **RECOMMENDED** Role for [Source Code Asset](#source-code-asset).                        |

<!-- lint disable no-undefined-references -->

> [!NOTE]
> (\*) These roles are offered as direct conversions from the previous extension
> that provided [ML-Model Asset Roles][ml-model-asset-roles] to provide easier upgrade to the MLM extension.

<!-- lint enable no-undefined-references -->

[ml-model-asset-roles]: https://github.com/stac-extensions/ml-model?tab=readme-ov-file#asset-objects

### Model Asset

| Field Name         | Type                                                                   | Description                                                                                                                                                                                                                                                                                                               |
|--------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| title              | string                                                                 | Description of the model asset.                                                                                                                                                                                                                                                                                           |
| href               | string                                                                 | URI to the model artifact.                                                                                                                                                                                                                                                                                                |
| type               | string                                                                 | The media type of the artifact (see [Model Artifact Media-Type](#model-artifact-media-type).                                                                                                                                                                                                                              |
| roles              | \[string]                                                              | **REQUIRED** Specify `mlm:model`. Can include `["mlm:weights", "mlm:checkpoint"]` as applicable.                                                                                                                                                                                                                          |
| mlm:artifact_type  | [Artifact Type](./best-practices.md#framework-specific-artifact-types) | Specifies the kind of model artifact, any string is allowed. Typically related to a particular ML framework, see [Best Practices - Framework Specific Artifact Types](./best-practices.md#framework-specific-artifact-types) for **RECOMMENDED** values. This field is **REQUIRED** if the `mlm:model` role is specified. |
| mlm:compile_method | [Compile Method](#compile-method) \| null                              | Describes the method used to compile the ML model either when the model is saved or at model runtime prior to inference.                                                                                                                                                                                                  |

Recommended Asset `roles` include `mlm:weights` or `mlm:checkpoint` for model weights that need to be loaded by a
model definition and `mlm:compiled` for models that can be loaded directly without an intermediate model definition.
In each case, the `mlm:model` **MUST** be applied as well to indicate that this asset represents the model.

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
See the [Best Practices - Framework Specific Artifact Types](./best-practices.md#framework-specific-artifact-types) on
 suggested fields for framework specific artifact types.

[iana-media-type]: https://www.iana.org/assignments/media-types/media-types.xhtml

[pytorch-aot-inductor]: https://pytorch.org/docs/main/torch.compiler_aot_inductor.html

#### Compile Method

| Compile Method | Description                                                                                                                                                                                                                                                                                                                                                                               |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| aot            | [Ahead-of-Time Compilation](https://en.wikipedia.org/wiki/Ahead-of-time_compilation). Converts a higher level code description of a model and a model's learned weights to a lower level representation prior to executing the model. This compiled model may be more portable by having fewer runtime dependencies and optimized for specific hardware.                                  |
| jit            | [Just-in-Time Compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation). Converts a higher level code description of a model and a model's learned weights to a lower level representation while executing the model. JIT provides more flexibility in the optimization approaches that can be applied to a model compared to AOT, but sacrifices portability and performance. |

### Source Code Asset

| Field Name     | Type      | Description                                                                                                               |
|----------------|-----------|---------------------------------------------------------------------------------------------------------------------------|
| title          | string    | Title of the source code.                                                                                                 |
| href           | string    | URI to the code repository, a ZIP archive, or an individual code/script file.                                             |
| type           | string    | Media-type of the URI.                                                                                                    |
| roles          | \[string] | **RECOMMENDED** Specify one or more of `["model", "code", "metadata"]`.                                                   |
| description    | string    | Description of the source code.                                                                                           |
| mlm:entrypoint | string    | Specific entrypoint reference in the code to use for running model inference. If specified, the `code` role is MANDATORY. |

If the referenced code does not directly offer a callable script to run the model
(i.e.: calling the script automatically resolves into invoking model inference within input data),
then the `mlm:entrypoint` field should be provided into an [Asset Object][stac-asset] in order to provide
a pointer to the inference function to execute the model.
For example, using a Python script, a `mlm:entrypoint` value of `"my_package.my_module:predict"` would refer
to the `predict` function located in the `my_module` inside the `my_package` library provided by the repository.

It is strongly recommended to use a specific media-type such as `text/x-python` if the source code refers directly
to a script of a known programming language. Using the HTML rendering of that source file, such as though GitHub
for example, should be avoided. It is preferable to provide the "Raw Contents" endpoint of the script to facilitate
its invocation without additional parsing to retrieve the source code.
The `text/html` media-type can be used for identification purpose using the URI referring to a version control 
repository such as Git, and where the `mlm:entrypoint` function can be easily resolved, through the corresponding 
package installed from PyPI. 

If an URI to a version control system is employed, it is recommended that it includes a specific commit hash, a
release number, a target branch or Git tag in order to ensure correct resolution of the specific model being described.
Embedded this information into the URI is recommended over other means of referring to checkout procedures simple to
facilitate the retrieval of the source code with a single reference, although this specification does not prohibit
the use of additional properties to better describe the Asset as needed.

Since the source code of a model provides useful example on how to use it, it is also recommended to define relevant
references to documentation using the `example` extension.
See the [Best Practices - Example Extension](best-practices.md#example-extension) section for more details.

Recommended asset `roles` include `code` and `metadata`,
since the source code asset might also refer to more detailed metadata than this specification captures.

### Container Asset

| Field Name | Type      | Description                                                                       |
|------------|-----------|-----------------------------------------------------------------------------------|
| title      | string    | Description of the container.                                                     |
| href       | string    | URI of the published container, including the container registry, image and tag.  |
| type       | string    | Media-type of the container, typically `application/vnd.oci.image.index.v1+json`. |
| roles      | \[string] | Specify `["runtime"]` and any other custom roles.                                 |

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
