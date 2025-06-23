# ML Model Extension Specification

<!-- lint disable no-undefined-references -->

> [!WARNING]
> This is legacy documentation reference of [ML-Model](https://github.com/stac-extensions/ml-model)
> preceding the current Machine Learning Model (MLM) extension.

<!-- lint enable no-undefined-references -->

## Migration Table

Following are the corresponding fields between the legacy DLM and the current MLM extension, which can be used to
completely migrate to the newer MLM extension providing enhanced feature and interconnectivity with other STAC
extensions (see also [Best Practices][mlm-bp]).

<!-- lint disable no-undefined-references -->

> [!NOTE]
> Only the limited set of `ml-model` fields are listed below for migration guidelines.
> See the full [MLM Specification](/../README.md) for all additional fields provided to further describe models.

<!-- lint enable no-undefined-references -->

### Item Properties

| ML-Model Field                                                 | MLM Field                                                                                                                                                          | Migration Details                                                                                                                                                                                                                                                                                                                                     |
|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ml-model:type` <br> (`"ml-model"` constant)                   | *n/a*                                                                                                                                                              | Including the MLM URI in `stac_extensions` is sufficient to indicate that the Item is a Model.                                                                                                                                                                                                                                                        |
| `ml-model:learning_approach`                                   | *n/a*                                                                                                                                                              | No direct mapping. Machine Learning training approaches can be very convoluted to describe. Instead, it is recommended to employ `derived_from` collection and other STAC Extension references to describe explicitly how the model was obtained. See [Best Practices][mlm-bp] for more details.                                                      |
| `ml-model:prediction_type` <br> (`string`)                     | `mlm:tasks` <br> (`[string]`)                                                                                                                                      | ML-Model limited to a single task. MLM allows multiple. Use `["<original-mlm-task>"]` to migrate directly.                                                                                                                                                                                                                                            |
| `ml-model:architecture`                                        | `mlm:architecture`                                                                                                                                                 | Direct mapping.                                                                                                                                                                                                                                                                                                                                       |
| `ml-model:training-processor-type` <br> `ml-model:training-os` | `mlm:framework` <br> `mlm:framework_version` <br> `mlm:accelerator` <br> `mlm:accelerator_constrained` <br> `mlm:accelerator_summary` <br> `mlm:accelerator_count` | More fields are provided to describe the subtleties of compute hardware and ML frameworks that can be intricated between them. If compute hardware imposes OS dependencies, they are typically reflected through the framework version and/or the specific accelerator. Further subtleties are permitted with [complex accelerator values][acc-type]. |

[acc-type]: /../README.md#accelerator-type-enum

[mlm-bp]: /../best-practices.md

### Asset Objects

#### Roles

All [ML-Model Asset Roles](https://github.com/stac-extensions/ml-model/blob/main/README.md#roles) 
are available with a prefix change with the same sematic meaning.

Further roles are also proposed in [MLM Asset Roles](/../README.md#mlm-asset-roles).

| ML-Model Field               | MLM Field               | Migration Details                                                                                                                      |
| ---------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `ml-model:inference-runtime` | `mlm:inference-runtime` | Prefix change.                                                                                                                         |
| `ml-model:training-runtime`  | `mlm:training-runtime`  | Prefix change.                                                                                                                         |
| `ml-model:checkpoint`        | `mlm:checkpoint`        | Prefix change. Recommended addition of further `mlm` properties for [Model Asset](/../README.md#model-asset) to describe the artifact. |

<!-- lint disable no-undefined-references -->

> [!NOTE]
> In the context of ML-Model, Assets providing [Inference/Training Runtimes][ml-model-runtimes]
> are strictly provided as [Docker Compose][docker-compose-file] definitions. While this is still permitted,
> the MLM extension offers alternatives using any relevant definition for the model, as long as it is properly
> identified by its applicable media-type. Additional recommendations and Asset property fields are provided
> under [MLM Assets Objects](/../README.md#assets-objects) for specific cases.

<!-- lint enable no-undefined-references -->

[ml-model-runtimes]: https://github.com/stac-extensions/ml-model/blob/main/README.md#inferencetraining-runtimes

[docker-compose-file]: https://github.com/compose-spec/compose-spec/blob/master/spec.md#compose-file

### Relation Types

| ML-Model Field                                  | MLM Field      | Migration Details                                                                                                                                                                                            |
| ----------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ml-model:inferencing-image`                    | *n/a*          | Deemed redundant with `mlm:inference-runtime` Asset Role.                                                                                                                                                    |
| `ml-model:training-image`                       | *n/a*          | Deemed redundant with `mlm:training-runtime` Asset Role.                                                                                                                                                     |
| `ml-model:train-data` <br> `ml-model:test-data` | `derived_from` | Use one or more `derived_from` links (as many as needed with regard to data involved during the model creation. Linked data should employ `ml-aoi` as appropriate (see [ML-AOI Best Practices][mlm-ml-aoi]). |

[mlm-ml-aoi]: /../README.md#ml-aoi-and-label-extensions
