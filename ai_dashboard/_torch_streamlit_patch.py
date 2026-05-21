"""Work around Streamlit inspecting torch.classes.__path__ (harmless console noise)."""

from __future__ import annotations

import logging


def apply_torch_streamlit_patch() -> None:
    """Run once before Streamlit starts its module path watcher."""
    try:
        import torch

        # Streamlit's watcher calls list(m.__path__._path); an empty path avoids
        # torch.classes.__getattr__ trying to resolve a fake '__path__._path' class.
        torch.classes.__path__ = []  # type: ignore[attr-defined]
    except Exception:
        pass

    class _TorchClassesLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return 'torch.classes' not in record.getMessage()

    log_filter = _TorchClassesLogFilter()
    for logger_name in (
        'streamlit',
        'streamlit.watcher',
        'streamlit.watcher.local_sources_watcher',
    ):
        logging.getLogger(logger_name).addFilter(log_filter)
