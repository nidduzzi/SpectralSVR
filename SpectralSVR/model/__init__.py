from .__base import (
    MultiRegression as MultiRegression,
    torch_json_encoder as torch_json_encoder,
    load_model as load_model,
    dump_model as dump_model,
    NumpyArrayorTensor as NumpyArrayorTensor,
)
from .SpectralSVR import SpectralSVR as SpectralSVR
from .LSSVR import LSSVR as LSSVR
from .FNN import FNN as FNN
