import pytest
import torch

HOST = torch.device("cpu")
ACCELERATOR_KIND = torch.accelerator.current_accelerator(check_available=True)
ACCELERATOR = (
    HOST
    if ACCELERATOR_KIND is None
    else torch.device(ACCELERATOR_KIND.type, torch.accelerator.current_device_index())
)
DEVICES_PRESENT = [HOST] + ([] if ACCELERATOR_KIND is None else [ACCELERATOR])
needs_an_accelerator = pytest.mark.skipif(
    ACCELERATOR_KIND is None, reason="needs an accelerator besides the host"
)
