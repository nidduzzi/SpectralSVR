"""Operator baselines: compare SpectralSVR against neural operators (DeepONet,
and later FNO/SNO/NSM) on the same problems, scored with the same metrics."""

from .__base import (
    FittedOperator as FittedOperator,
    Operator as Operator,
)
from .benchmark import benchmark as benchmark
from .dataset import (
    OperatorDataset as OperatorDataset,
    Split as Split,
    sensor_grid as sensor_grid,
)
from .deeponet import (
    DeepONetOperator as DeepONetOperator,
    FittedDeepONet as FittedDeepONet,
    DeepONetRecipe as DeepONetRecipe,
    Lu2021DeepxdeExample as Lu2021DeepxdeExample,
    Lu2022Burgers as Lu2022Burgers,
)
from .spectralsvr import (
    FittedSpectralSVR as FittedSpectralSVR,
    SpectralSVROperator as SpectralSVROperator,
)
