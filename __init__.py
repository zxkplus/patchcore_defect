"""PatchCore Defect - 带缺陷特征库的目标检测与分割系统"""

__version__ = "1.0.0"

from patchcore_defect.src.patchcore.patchcore import PatchCore
from patchcore_defect.src.patchcore.defect_library import DefectFeatureLibrary
from patchcore_defect.src.patchcore.dual_scorer import DualLibraryScorer
from patchcore_defect.src.patchcore.datasets import DefectDataset, MVTecDefectDataset

__all__ = [
    "PatchCore",
    "DefectFeatureLibrary", 
    "DualLibraryScorer",
    "DefectDataset",
    "MVTecDefectDataset",
]
