import pytest


@pytest.fixture
def metadata_json():
    from stac_model.examples import eurosat_resnet

    model_metadata_stac_item = eurosat_resnet()
    return model_metadata_stac_item


def test_model_metadata_to_dict(metadata_json):
    assert metadata_json.to_dict()


def test_model_metadata_json_operations(metadata_json):
    from stac_model.schema import MLModelExtension

    assert MLModelExtension(metadata_json.to_dict())
