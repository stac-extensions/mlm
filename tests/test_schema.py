import pystac


def test_mlm_schema(mlm_validator, mlm_example):
    mlm_item = pystac.Item.from_dict(mlm_example)
    invalid = pystac.validation.validate(mlm_item, validator=mlm_validator)
    assert not invalid


def test_model_metadata_to_dict(eurosat_resnet):
    assert eurosat_resnet.item.to_dict()


def test_validate_model_metadata(eurosat_resnet):
    assert pystac.read_dict(eurosat_resnet.item.to_dict())
