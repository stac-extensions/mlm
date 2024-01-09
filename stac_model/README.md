# stac-model

<div align="center">

[![Python support][bp1]][bp2]
[![PyPI Release][bp3]][bp2]
[![Repository][bscm1]][bp4]
[![Releases][bscm2]][bp5]
[![Docs][bdoc1]][bdoc2]

[![Contributions Welcome][bp8]][bp9]

[![Poetry][bp11]][bp12]
[![Pre-commit][bp15]][bp16]
[![Semantic versions][blic3]][bp5]
[![Pipelines][bscm6]][bscm7]

_A PydanticV2 validation and serialization library for the STAC ML Model Extension_

</div>

## Installation

```bash
pip install -U stac-model
```

or install with `Poetry`:

```bash
poetry add stac-model
```
Then you can run

```bash
stac-model --help
```

or with `Poetry`:

```bash
poetry run stac-model --help
```

## Creating an example metadata json

```
poetry run stac-model
```

This will make an example example.json metadata file for an example model.

Currently this looks like

```
{
  "mlm_input": {
    "name": "13 Band Sentinel-2 Batch",
    "bands": [],
    "input_array": {
      "shape": [
        -1,
        13,
        64,
        64
      ],
      "dim_order": "bchw",
      "dtype": "float32"
    },
    "norm_type": "z_score",
    "rescale_type": "none",
    "norm_by_channel": true,
    "statistics": {
      "mean": [
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798
      ],
      "stddev": [
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.4170017,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042
      ]
    },
    "pre_processing_function": "https://github.com/microsoft/torchgeo/blob/545abe8326efc2848feae69d0212a15faba3eb00/torchgeo/datamodules/eurosat.py"
  },
  "mlm_architecture": {
    "name": "ResNet-18",
    "summary": "Sourced from torchgeo python library, identifier is ResNet18_Weights.SENTINEL2_ALL_MOCO",
    "pretrained": true,
    "total_parameters": 11700000
  },
  "mlm_runtime": {
    "framework": "torch",
    "version": "2.1.2+cu121",
    "asset": {
      "href": "."
    },
    "source_code_url": "https://github.com/huggingface/pytorch-image-models/blob/b5a4fa9c3be6ac732807db7e87d176f5e4fc06f1/timm/models/resnet.py#L362",
    "handler": "torchgeo.models.resnet.ResNet18"
  },
  "mlm_output": {
    "task": "classification",
    "number_of_classes": 10,
    "output_shape": [
      -1,
      10
    ],
    "class_name_mapping": {
      "Annual Crop": 0,
      "Forest": 1,
      "Herbaceous Vegetation": 2,
      "Highway": 3,
      "Industrial Buildings": 4,
      "Pasture": 5,
      "Permanent Crop": 6,
      "Residential Buildings": 7,
      "River": 8,
      "SeaLake": 9
    }
  }
}
```

## :chart_with_upwards_trend: Releases

You can see the list of available releases on the [GitHub Releases][r1] page.

## :page_facing_up:  License
[![License][blic1]][blic2]

This project is licenced under the terms of the `Apache Software License 2.0` licence. See [LICENSE][blic2] for more details.

## :heartpulse: Credits
[![Python project templated from galactipy.][bp6]][bp7]

<!-- Anchors -->

[bp1]: https://img.shields.io/pypi/pyversions/stac-model?style=for-the-badge
[bp2]: https://pypi.org/project/stac-model/
[bp3]: https://img.shields.io/pypi/v/stac-model?style=for-the-badge&logo=pypi&color=3775a9
[bp4]: https://github.com/stac-extensions/stac-model
[bp5]: https://github.com/stac-extensions/stac-model/releases
[bp6]: https://img.shields.io/badge/made%20with-galactipy%20%F0%9F%8C%8C-179287?style=for-the-badge&labelColor=193A3E
[bp7]: https://kutt.it/7fYqQl
[bp8]: https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=for-the-badge
[bp9]: https://github.com/stac-extensions/stac-model/blob/main/CONTRIBUTING.md
[bp11]: https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json&style=for-the-badge
[bp12]: https://python-poetry.org/

[bp15]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge
[bp16]: https://github.com/stac-extensions/stac-model/blob/main/.pre-commit-config.yaml

[blic1]: https://img.shields.io/github/license/stac-extensions/stac-model?style=for-the-badge
[blic2]: https://github.com/stac-extensions/stac-model/blob/main/LICENCE
[blic3]: https://img.shields.io/badge/%F0%9F%93%A6-semantic%20versions-4053D6?style=for-the-badge

[r1]: https://github.com/stac-extensions/stac-model/releases

[bscm1]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[bscm2]: https://img.shields.io/github/v/release/stac-extensions/stac-model?style=for-the-badge&logo=semantic-release&color=347d39
[bscm6]: https://img.shields.io/github/actions/workflow/status/stac-extensions/stac-model/build.yml?style=for-the-badge&logo=github
[bscm7]: https://github.com/stac-extensions/stac-model/actions/workflows/build.yml

[hub1]: https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuring-dependabot-version-updates#enabling-dependabot-version-updates
[hub2]: https://github.com/marketplace/actions/close-stale-issues
[hub5]: https://github.com/stac-extensions/stac-model/blob/main/.github/workflows/build.yml
[hub6]: https://docs.github.com/en/code-security/dependabot
[hub8]: https://github.com/stac-extensions/stac-model/blob/main/.github/release-drafter.yml
[hub9]: https://github.com/stac-extensions/stac-model/blob/main/.github/.stale.yml

[bdoc1]: https://img.shields.io/badge/docs-github%20pages-0a507a?style=for-the-badge
[bdoc2]: https://stac-extensions.github.io/stac-model
