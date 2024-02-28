
import pytest


@pytest.fixture
def metadata_json():
    from stac_model.examples import eurosat_resnet
    model_metadata_stac_item = eurosat_resnet()
    return model_metadata_stac_item

def test_model_metadata_json_operations(model_metadata_stac_item):
    from stac_model.schema import MLModelExtension
    model_metadata = MLModelExtension.apply(model_metadata_stac_item)
    assert model_metadata.mlm_name == "Resnet-18 Sentinel-2 ALL MOCO"
