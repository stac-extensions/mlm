from datetime import datetime

import pystac

from stac_model.schema import (
    Asset,
    ClassObject,
    InputArray,
    MLModelExtension,
    MLModelProperties,
    ModelInput,
    ModelOutput,
    ResultArray,
    Runtime,
    Statistics,
)


def eurosat_resnet():
    input_array = InputArray(
        shape=[-1, 13, 64, 64], dim_order="bchw", data_type="float32"
    )
    band_names = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
    stats = Statistics(
        mean=[
            1354.40546513,
            1118.24399958,
            1042.92983953,
            947.62620298,
            1199.47283961,
            1999.79090914,
            2369.22292565,
            2296.82608323,
            732.08340178,
            12.11327804,
            1819.01027855,
            1118.92391149,
            2594.14080798,
        ],
        stddev=[
            245.71762908,
            333.00778264,
            395.09249139,
            593.75055589,
            566.4170017,
            861.18399006,
            1086.63139075,
            1117.98170791,
            404.91978886,
            4.77584468,
            1002.58768311,
            761.30323499,
            1231.58581042,
        ],
    )
    mlm_input = ModelInput(
        name="13 Band Sentinel-2 Batch",
        bands=band_names,
        input_array=input_array,
        norm_by_channel=True,
        norm_type="z_score",
        resize_type="none",
        statistics=stats,
        pre_processing_function="https://github.com/microsoft/torchgeo/blob/545abe8326efc2848feae69d0212a15faba3eb00/torchgeo/datamodules/eurosat.py",  # noqa: E501
    )
    mlm_runtime = Runtime(
        framework="torch",
        version="2.1.2+cu121",
        asset=Asset(
            href="https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco/resolve/main/resnet18_sentinel2_all_moco-59bfdff9.pth"  # noqa: E501
        ),
        source_code=Asset(
            href="https://github.com/microsoft/torchgeo/blob/61efd2e2c4df7ebe3bd03002ebbaeaa3cfe9885a/torchgeo/models/resnet.py#L207"  # noqa: E501
        ),
        accelerator="cuda",
        accelerator_constrained=False,
        hardware_summary="Unknown",
    )
    result_array = ResultArray(
        shape=[-1, 10], dim_names=["batch", "class"], data_type="float32"
    )
    class_map = {
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
    class_objects = [
        ClassObject(value=class_map[class_name], name=class_name)
        for class_name in class_map
    ]
    mlm_output = ModelOutput(
        task="classification",
        classification_classes=class_objects,
        output_shape=[-1, 10],
        result_array=[result_array],
    )
    ml_model_meta = MLModelProperties(
        mlm_name="Resnet-18 Sentinel-2 ALL MOCO",
        mlm_task="classification",
        mlm_framework="pytorch",
        mlm_framework_version="2.1.2+cu121",
        mlm_file_size=1,
        mlm_memory_size=1,
        mlm_summary=(
            "Sourced from torchgeo python library,"
            "identifier is ResNet18_Weights.SENTINEL2_ALL_MOCO"
        ),
        mlm_pretrained_source="EuroSat Sentinel-2",
        mlm_total_parameters=11_700_000,
        mlm_input=[mlm_input],
        mlm_runtime=[mlm_runtime],
        mlm_output=[mlm_output],
    )
    start_datetime = datetime.strptime("1900-01-01", "%Y-%m-%d")
    end_datetime = None
    geometry = None
    bbox = [-90, -180, 90, 180]
    name = (
        "_".join(ml_model_meta.mlm_name.split(" ")).lower()
        + f"_{ml_model_meta.mlm_task}".lower()
    )
    item = pystac.Item(
        id=name,
        geometry=geometry,
        bbox=bbox,
        datetime=None,
        properties={"start_datetime": start_datetime, "end_datetime": end_datetime},
    )
    item.add_derived_from(
        "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a"
    )
    item_mlmodel = MLModelExtension.ext(item, add_if_missing=True)
    item_mlmodel.apply(ml_model_meta.model_dump())
    return item_mlmodel
