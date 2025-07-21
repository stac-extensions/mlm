import json
from pathlib import Path

from stac_model.examples import unet_mlm


def test_unet_mlm_matches_example_json():
    json_path = Path(__file__).parent.parent / "examples" / "item_pytorch_geo_unet.json"
    with open(json_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    item = unet_mlm().item.to_dict()

    assert item == expected, "Generated STAC Item does not match the saved example."
