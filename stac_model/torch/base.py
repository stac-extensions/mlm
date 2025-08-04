from typing_extensions import TypedDict

import torch

from ..base import Paths

ExtraFiles = TypedDict("ExtraFiles", {"mlm-metadata": str}, total=False)


class ExportedPrograms(TypedDict, total=False):
    model: torch.export.ExportedProgram
    transforms: torch.export.ExportedProgram


class AOTIFiles(TypedDict, total=False):
    model: Paths
    transforms: Paths
