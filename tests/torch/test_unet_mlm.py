import json
from pathlib import Path

import pytest

pytest.importorskip("torchgeo")

from stac_model.examples import unet_mlm


@pytest.fixture(autouse=True)
def validated_torchgeo_unet_mlm():  # pragma: has-torchgeo-unet
    """Ensure that the test is running against the expected torchgeo version and model definition."""
    item = unet_mlm().item.to_dict()

    # bands not the same depending on torchgeo version
    # https://github.com/torchgeo/torchgeo/commit/41411d4511e0bd1b135e5ba77af1401d0ee0c6e7
    from torchgeo.models.unet import _ftw_sentinel2_bands
    assert _ftw_sentinel2_bands == ["B4", "B3", "B2", "B8", "B4", "B3", "B2", "B8"]

    # this part is manually added in the reference example for documentation purpose
    item.update({
        "$comment": (
            "STAC item auto-generated using unet_mlm() in "
            "https://raw.githubusercontent.com/stac-extensions/mlm/refs/heads/main/stac_model/examples.py"
        )
    })

    return item


def test_unet_mlm_matches_example_json(validated_torchgeo_unet_mlm):  # pragma: has-torchgeo-unet
    json_path = Path(__file__).resolve().parents[2] / "examples" / "item_pytorch_geo_unet.json"
    with open(json_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    assert validated_torchgeo_unet_mlm == expected, "Generated STAC Item does not match the saved example."
