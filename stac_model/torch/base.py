from typing import NotRequired, TypedDict

import torch

from ..base import Paths

ExtraFiles = TypedDict("ExtraFiles", {"mlm-metadata": str})

class ExportedPrograms(TypedDict, total=False):
    model: torch.export.ExportedProgram
    transforms: NotRequired[torch.export.ExportedProgram]


class AOTIFiles(TypedDict, total=False):
    model: Paths
    transforms: NotRequired[Paths]
