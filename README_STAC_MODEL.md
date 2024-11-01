# stac-model

<!--lint disable no-html -->

<div align="center">

[![Python support][bp1]][bp2]
[![PyPI Release][bp3]][bp2]
[![Repository][bscm1]][bp4]
[![Releases][bscm2]][bp5]
[![Docs][bdoc1]][bdoc2]

[![Contributions Welcome][bp8]][bp9]

[![uv][bp11]][bp12]
[![Pre-commit][bp15]][bp16]
[![Semantic versions][blic3]][bp5]
[![Pipelines][bscm6]][bscm7]

_A PydanticV2 and PySTAC validation and serialization library for the STAC ML Model Extension_

</div>

> ⚠️ <br>
> FIXME: update description with ML framework connectors (pytorch, scikit-learn, etc.)

## Installation

```shell
pip install -U stac-model
```

or install with `uv`:

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
[bp11]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
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

[bdoc1]: https://img.shields.io/badge/docs-github%20pages-0a507a?style=for-the-badge
[bdoc2]: https://github.com/stac-extensions/mlm/blob/main/README_STAC_MODEL.md
