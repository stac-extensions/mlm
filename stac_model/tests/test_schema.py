import pytest
from stac_model.schema import (
    TensorSignature,
    ModelSignature,
    ModelArtifact,
    ClassMap,
    ModelMetadata,
)
import os
import tempfile


def create_metadata():
    input_sig = TensorSignature(
        name="input_tensor", dtype="float32", shape=(-1, 13, 64, 64)
    )
    output_sig = TensorSignature(name="output_tensor", dtype="float32", shape=(-1, 10))
    model_sig = ModelSignature(inputs=[input_sig], outputs=[output_sig])
    model_artifact = ModelArtifact(path="s3://example/s3/uri/model.pt")
    class_map = ClassMap(
        class_to_label_id={
            "Annual Crop": 0,
            "Forest": 1,
            "Herbaceous Vegetation": 2,
            "Highway": 3,
            "Industrial Buildings": 4,
            "Pasture": 5,
            "Permanent Crop": 6,
            "Residential Buildings": 7,
            "River": 8,
            "SeaLake": 9,
        }
    )
    return ModelMetadata(
        name="eurosat",
        class_map=class_map,
        signatures=model_sig,
        artifact=model_artifact,
        ml_model_processor_type="cpu",
    )


@pytest.fixture
def metadata_json():
    model_metadata = create_metadata()
    return model_metadata.model_dump_json(indent=2)


def test_model_metadata_json_operations(metadata_json):
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, "tempfile.json")

        # Write to the file
        with open(temp_filepath, "w") as file:
            file.write(metadata_json)

        # Read and validate the model metadata from the JSON file
        with open(temp_filepath, "r") as json_file:
            json_str = json_file.read()
            model_metadata = ModelMetadata.model_validate_json(json_str)

    assert model_metadata.name == "eurosat"


def test_benchmark_model_metadata_validation(benchmark):
    json_str = create_metadata().model_dump_json(indent=2)
    benchmark(ModelMetadata.model_validate_json, json_str)
