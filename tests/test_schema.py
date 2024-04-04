import pytest


@pytest.fixture
def mlmodel_metadata_item():
    from stac_model.examples import eurosat_resnet

    model_metadata_stac_item = eurosat_resnet()
    return model_metadata_stac_item


def test_model_metadata_to_dict(mlmodel_metadata_item):
    assert mlmodel_metadata_item.item.to_dict()


def test_validate_model_metadata(mlmodel_metadata_item):
    import pystac
    assert pystac.read_dict(mlmodel_metadata_item.item.to_dict())
