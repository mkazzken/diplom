"""Lazy PyTorch import and Streamlit compatibility patch."""
from __future__ import annotations

torch = None
TemporalUNet = None
InspiredSurrogate = None
TORCH_AVAILABLE = None
TORCH_ERROR = None


def ensure_torch() -> bool:
    global torch, TemporalUNet, InspiredSurrogate, TORCH_AVAILABLE, TORCH_ERROR
    if TORCH_AVAILABLE is not None:
        return TORCH_AVAILABLE

    try:
        from ._torch_streamlit_patch import apply_torch_streamlit_patch

        apply_torch_streamlit_patch()

        import torch as torch_module
        from unet import TemporalUNet as PureUNet
        from hubrid_unet import InspiredSurrogate as HybridSurrogate

        torch = torch_module
        TemporalUNet = PureUNet
        InspiredSurrogate = HybridSurrogate
        TORCH_AVAILABLE = True
        TORCH_ERROR = None
    except Exception as exc:
        torch = None
        TemporalUNet = None
        InspiredSurrogate = None
        TORCH_AVAILABLE = False
        TORCH_ERROR = str(exc)

    return TORCH_AVAILABLE
