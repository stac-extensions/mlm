import os
import tempfile

import pytest


@pytest.fixture
def metadata_json():
    from stac_model.examples import eurosat_resnet
    model_metadata = eurosat_resnet()
    return model_metadata.model_dump_json(indent=2)

def test_model_metadata_json_operations(metadata_json):
    from stac_model.schema import MLModel
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, "tempfile.json")
        with open(temp_filepath, "w") as file:
            file.write(metadata_json)
        with open(temp_filepath) as json_file:
            json_str = json_file.read()
            model_metadata = MLModel.model_validate_json(json_str)
    assert model_metadata.name == "Resnet-18 Sentinel-2 ALL MOCO"
