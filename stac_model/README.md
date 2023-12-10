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
  "signatures": {
    "inputs": [
      {
        "name": "input_tensor",
        "dtype": "float32",
        "shape": [
          -1,
          13,
          64,
          64
        ]
      }
    ],
    "outputs": [
      {
        "name": "output_tensor",
        "dtype": "float32",
        "shape": [
          -1,
          10
        ]
      }
    ],
    "params": null
  },
  "artifact": {
    "path": "s3://example/s3/uri/model.pt",
    "additional_files": null
  },
  "id": "3fa03dceb4004b6e8a9e8591e4b3a99d",
  "class_map": {
    "class_to_label_id": {
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
  },
  "runtime_config": null,
  "name": "eurosat",
  "ml_model_type": null,
  "ml_model_processor_type": "cpu",
  "ml_model_learning_approach": null,
  "ml_model_prediction_type": null,
  "ml_model_architecture": null
}
```

## :chart_with_upwards_trend: Releases

You can see the list of available releases on the [GitHub Releases][r1] page.


[![License][blic1]][blic2]

This project is licenced under the terms of the `Apache Software License 2.0` licence. See [LICENCE][blic2] for more details.


## Credits [![Python project templated from galactipy.][bp6]][bp7]

This project was generated with [`galactipy`][bp7].

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
