from typing import Annotated, Any, Dict, List, Optional, Set, Union, cast

from pydantic import AliasChoices, ConfigDict, Field, model_serializer
from pystac.extensions.classification import Classification

from stac_model.base import DataType, MLMBaseModel, ModelTask, OmitIfNone, ProcessingExpression


class ModelResult(MLMBaseModel):
    shape: List[Union[int, float]] = Field(..., min_items=1)
    dim_order: List[str] = Field(..., min_items=1)
    data_type: DataType


# MLMClassification: TypeAlias = Annotated[
#     Classification,
#     PlainSerializer(
#         lambda x: x.to_dict(),
#         when_used="json",
#         return_type=TypedDict(
#             "Classification",
#             {
#                 "value": int,
#                 "name": str,
#                 "description": NotRequired[str],
#                 "color_hint": NotRequired[str],
#             }
#         )
#     )
# ]


class MLMClassification(MLMBaseModel, Classification):
    @model_serializer()
    def model_dump(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return self.to_dict()  # type: ignore[call-arg]

    def __init__(
        self,
        value: int,
        description: Optional[str] = None,
        name: Optional[str] = None,
        color_hint: Optional[str] = None,
    ) -> None:
        Classification.__init__(self, {})
        if not name and not description:
            raise ValueError("Class name or description is required!")
        self.apply(
            value=value,
            name=name or description,
            description=cast(str, description or name),
            color_hint=color_hint,
        )

    def __hash__(self) -> int:
        return sum(map(hash, self.to_dict().items()))

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "properties":
            Classification.__setattr__(self, key, value)
        else:
            MLMBaseModel.__setattr__(self, key, value)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


# class ClassObject(BaseModel):
#     value: int
#     name: str
#     description: Optional[str] = None
#     title: Optional[str] = None
#     color_hint: Optional[str] = None
#     nodata: Optional[bool] = False


class ModelOutput(MLMBaseModel):
    name: str
    tasks: Set[ModelTask]
    result: ModelResult

    # NOTE:
    #   Although it is preferable to have 'Set' to avoid duplicate,
    #   it is more important to keep the order in this case,
    #   which we would lose with 'Set'.
    #   We also get some unhashable errors with 'Set', although 'MLMClassification' implements '__hash__'.
    classes: Annotated[List[MLMClassification], OmitIfNone] = Field(
        alias="classification:classes",
        validation_alias=AliasChoices("classification:classes", "classification_classes", "classes"),
    )
    post_processing_function: Optional[ProcessingExpression] = None

    model_config = ConfigDict(
        populate_by_name=True,
    )
