# stac-model

<!--lint disable no-html -->

<div align="center">

[![Python support][bp1]][bp2]
[![PyPI Release][bp3]][bp2]
[![Repository][bscm1]][bp4]
[![Releases][bscm2]][bp5]

[![Contributions Welcome][bp8]][bp9]

[![uv][bp11]][bp12]
[![Pre-commit][bp15]][bp16]
[![Semantic versions][blic3]][bp5]
[![Pipelines][bscm6]][bscm7]

*A PydanticV2 and PySTAC validation and serialization library for the STAC ML Model Extension*

</div>

> ⚠️ <br>
> FIXME: update description with ML framework connectors (pytorch, scikit-learn, etc.)

## Installation

```shell
pip install -U stac-model
```

or install with uv:

```shell
uv add stac-model
```

Then you can run

```shell
stac-model --help
```

## Creating example metadata JSON for a STAC Item

```shell
stac-model
```

This will make [this example item](./examples/item_basic.json) for an example model.

## Validating Model Metadata

An alternative use of `stac_model` is to validate config files containing model metadata using the `MLModelProperties` schema.

Given a YAML or JSON file with the structure in [examples/torch/mlm-metadata.yaml](./examples/torch/mlm-metadata.yaml), the model metadata can be validated as follows:

```python
import yaml
from stac_model.schema import MLModelProperties

with open("examples/mlm-metadata.yaml", "r", encoding="utf-8") as f:
    metadata = yaml.safe_load(f)

MLModelProperties.model_validate(metadata["properties"])  
```

## Exporting and Packaging PyTorch Models, Transforms, and Model Metadata

As of PyTorch 2.8, and stac_model 1.5.0, you can now export and package PyTorch models, transforms,
and model metadata using functions in `stac_model.torch.export`. Below is an example of exporting a
U-Net model pretrained on the [Fields of The World (FTW) dataset](https://fieldsofthe.world/) for
field boundary segmentation in Sentinel-2 satellite imagery using the [TorchGeo](https://github.com/microsoft/torchgeo) library.

> 📝 **Note:** To customize the metadata for your model you can use this [example](./tests/torch/metadata.yaml) as a template.

```python
import torch
import torchvision.transforms.v2 as T
from torchgeo.models import Unet_Weights, unet
from stac_model.torch.export import save

weights = Unet_Weights.SENTINEL2_3CLASS_FTW
transforms = torch.nn.Sequential(
  T.Resize((256, 256)),
  T.Normalize(mean=[0.0], std=[3000.0])
)
model = unet(weights=weights)

save(
    output_file="ftw.pt2",
    model=model,  # Must be an nn.Module
    transforms=transforms,  # Must be an nn.Module
    metadata_path="metadata.yaml",  # Can be a metadata yaml or MLModelProperties object
    input_shape=[-1, 8, -1, -1],  # -1 indicates a dynamic shaped dimension
    device="cpu",
    dtype=torch.float32,
    aoti_compile_and_package=False,  # True for AOTInductor compile otherwise use torch.export
)
```

The model, transforms, and metadata can then be loaded into an environment with only torch and stac_model as required dependencies like below:

```python
import yaml
from torch.export.pt2_archive._package import load_pt2

pt2 = load_pt2(archive_path)
metadata = yaml.safe_load(pt2.extra_files["mlm-metadata"])

# If exported with aoti_compile_and_package=True
model = pt2.aoti_runners["model"]
transforms = pt2.aoti_runners["transforms"]

# If exported with aoti_compile_and_package=False
model = pt2.exported_programs["model"].module()
transforms = pt2.exported_programs["transforms"].module()

# Inference
batch = ...  # An input batch tensor
outputs = model(transforms(batch))
```

## 📈 Releases

You can see the list of available releases on the [GitHub Releases][github-releases] page.

## 📄 License

[![License][blic1]][blic2]

This project is licenced under the terms of the `Apache Software License 2.0` licence.
See [LICENSE][blic2] for more details.

## 💗 Credits

[![Python project templated from galactipy.][bp6]][bp7]

<!-- Anchors -->

[bp1]: https://img.shields.io/pypi/pyversions/stac-model?style=for-the-badge

[bp2]: https://pypi.org/project/stac-model/

[bp3]: https://img.shields.io/pypi/v/stac-model?style=for-the-badge&logo=pypi&color=3775a9

[bp4]: https://github.com/stac-extensions/mlm

[bp5]: https://github.com/stac-extensions/mlm/releases

[bp6]: https://img.shields.io/badge/made%20with-galactipy%20%F0%9F%8C%8C-179287?style=for-the-badge&labelColor=193A3E

[bp7]: https://kutt.it/7fYqQl

[bp8]: https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=for-the-badge

[bp9]: https://github.com/stac-extensions/mlm/blob/main/CONTRIBUTING.md

[bp11]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge

[bp12]: https://docs.astral.sh/uv/

[bp15]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge

[bp16]: https://github.com/stac-extensions/mlm/blob/main/.pre-commit-config.yaml

[blic1]: https://img.shields.io/github/license/stac-extensions/mlm?style=for-the-badge

[blic2]: https://github.com/stac-extensions/mlm/blob/main/LICENSE

[blic3]: https://img.shields.io/badge/%F0%9F%93%A6-semantic%20versions-4053D6?style=for-the-badge

[github-releases]: https://github.com/stac-extensions/mlm/releases

[bscm1]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white

[bscm2]: https://img.shields.io/github/v/release/stac-extensions/mlm?filter=stac-model-v*&style=for-the-badge&logo=semantic-release&color=347d39

[bscm6]: https://img.shields.io/github/actions/workflow/status/stac-extensions/mlm/publish.yaml?style=for-the-badge&logo=github

[bscm7]: https://github.com/stac-extensions/mlm/blob/main/.github/workflows/publish.yaml

[hub1]: https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuring-dependabot-version-updates#enabling-dependabot-version-updates

[hub2]: https://github.com/marketplace/actions/close-stale-issues

[hub6]: https://docs.github.com/en/code-security/dependabot

[hub8]: https://github.com/stac-extensions/mlm/blob/main/.github/release-drafter.yml

[hub9]: https://github.com/stac-extensions/mlm/blob/main/.github/.stale.yml
