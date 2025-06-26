# ML Model Extension Specification

<!-- lint disable no-undefined-references -->

> [!WARNING]
> This is legacy documentation reference of [ML-Model][ml-model]
> preceding the current Machine Learning Model ([MLM][mlm-spec]) extension.

<!-- lint enable no-undefined-references -->

## Notable Differences

- The [MLM][mlm-spec] Extension covers more details at both the [Item](#item-properties) and [Asset](#asset-objects)
  levels, making it easier to describe and use model metadata.
- The [MLM][mlm-spec] extension covers Runtime requirements within the [Container Asset](./../../README.md#container-asset),
  while the [ML-Model][ml-model] extension records [similar information][ml-model-runtimes] in
  the `ml-model:inference-runtime` or `ml-model:training-runtime` [Asset Roles](#roles).
- The [MLM][mlm-spec] extension has a corresponding Python library, [`stac-model`](https://pypi.org/project/stac-model/)
  which can be used to create and validate MLM metadata.
  An example of the library in action is [provided in examples](./../../stac_model/examples.py).
  The [ML-Model][ml-model] extension does not support this and requires the JSON to be written manually
  by interpreting the JSON Schema or existing examples.
- The [MLM][mlm-spec] extension is easier to maintain and enhance in a fast moving ML ecosystem thanks to it's use
  of [pydantic][pydantic] models, while still being compatible with [pystac][pystac] for extension and STAc core
  validation.

## Migration Tables

Following are the corresponding fields between the legacy [ML-Model][ml-model] and the current [MLM][mlm-spec]
extension, which can be used to completely migrate to the newer *Machine Leaning Model* extension providing
enhanced features and interconnectivity with other STAC extensions (see also [Best Practices][mlm-bp]).

<!-- lint disable no-undefined-references -->

> [!IMPORTANT]
> Only the limited set of [`ml-model`][ml-model] fields are listed below for migration guidelines.
> See the full [MLM Specification](./../../README.md) for all additional fields provided to further describe models.

<!-- lint enable no-undefined-references -->

### Item Properties

| ML-Model Field                                                 | MLM Field                                                                                                                                                          | Migration Details                                                                                                                                                                                                                                                                                                                                         |
|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ml-model:type` <br> (`"ml-model"` constant)                   | *n/a*                                                                                                                                                              | Including the MLM URI in `stac_extensions` is sufficient to indicate that the Item is a model.                                                                                                                                                                                                                                                            |
| `ml-model:learning_approach`                                   | *n/a*                                                                                                                                                              | No direct mapping. Machine Learning training approaches can be very convoluted to describe. Instead, it is recommended to employ `derived_from` collection and other STAC Extension references to describe explicitly how the model was obtained. See [Best Practices][mlm-bp] for more details.                                                          |
| `ml-model:prediction_type` <br> (`string`)                     | `mlm:tasks` <br> (`[string]`)                                                                                                                                      | ML-Model limited to a single task. MLM allows multiple. Use `["<original-mlm-task>"]` to migrate directly.                                                                                                                                                                                                                                                |
| `ml-model:architecture`                                        | `mlm:architecture`                                                                                                                                                 | Direct mapping.                                                                                                                                                                                                                                                                                                                                           |
| `ml-model:training-processor-type` <br> `ml-model:training-os` | `mlm:framework` <br> `mlm:framework_version` <br> `mlm:accelerator` <br> `mlm:accelerator_constrained` <br> `mlm:accelerator_summary` <br> `mlm:accelerator_count` | More fields are provided to describe the subtleties of compute hardware and ML frameworks that can be intricated between them. If compute hardware imposes OS dependencies, they are typically reflected through the framework version and/or the specific accelerator. Further subtleties are permitted with [complex accelerator values][mlm-acc-type]. |

### Asset Objects

#### Roles

All [ML-Model Asset Roles](https://github.com/stac-extensions/ml-model/blob/main/README.md#roles) 
are available with a prefix change with the same sematic meaning.

Further roles are also proposed in [MLM Asset Roles](./../../README.md#mlm-asset-roles).

| ML-Model Field               | MLM Field               | Migration Details                                                                                                                          |
|------------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `ml-model:inference-runtime` | `mlm:inference-runtime` | Prefix change.                                                                                                                             |
| `ml-model:training-runtime`  | `mlm:training-runtime`  | Prefix change.                                                                                                                             |
| `ml-model:checkpoint`        | `mlm:checkpoint`        | Prefix change. Recommended addition of further `mlm` properties for [Model Asset](./../../README.md#model-asset) to describe the artifact. |

<!-- lint disable no-undefined-references -->

> [!TIP]
> In the context of [ML-Model][ml-model], Assets providing [Inference/Training Runtimes][ml-model-runtimes]
> are strictly provided as [Docker Compose][docker-compose-file] definitions. While this is still permitted,
> the MLM extension offers alternatives using any relevant definition for the model, as long as it is properly
> identified by its applicable media-type. Additional recommendations and Asset property fields are provided
> under [MLM Assets Objects](./../../README.md#assets-objects) for specific cases.

<!-- lint enable no-undefined-references -->

### Relation Types

| ML-Model Field                                  | MLM Field      | Migration Details                                                                                                                                                                                            |
|-------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ml-model:inferencing-image`                    | *n/a*          | Deemed redundant with `mlm:inference-runtime` Asset Role.                                                                                                                                                    |
| `ml-model:training-image`                       | *n/a*          | Deemed redundant with `mlm:training-runtime` Asset Role.                                                                                                                                                     |
| `ml-model:train-data` <br> `ml-model:test-data` | `derived_from` | Use one or more `derived_from` links (as many as needed with regard to data involved during the model creation. Linked data should employ `ml-aoi` as appropriate (see [ML-AOI Best Practices][mlm-ml-aoi]). |

[mlm-acc-type]: ./../../README.md#accelerator-type-enum

[mlm-spec]: ./../../README.md

[mlm-bp]: ./../../best-practices.md

[mlm-ml-aoi]: ./../../best-practices.md#ml-aoi-and-label-extensions

[ml-model]: https://github.com/stac-extensions/ml-model

[ml-model-runtimes]: https://github.com/stac-extensions/ml-model/blob/main/README.md#inferencetraining-runtimes

[pydantic]: https://docs.pydantic.dev/latest/

[pystac]: https://pystac.readthedocs.io/en/latest/

[docker-compose-file]: https://github.com/compose-spec/compose-spec/blob/master/spec.md#compose-file
