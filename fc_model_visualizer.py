from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parent / "lectures" / "05-neural-networks" / "fc_model_visualizer.py"
_SPEC = importlib.util.spec_from_file_location("_fc_model_visualizer_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load fully connected model visualizer from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

LinearLayerSpec = _MODULE.LinearLayerSpec
build_demo_model = _MODULE.build_demo_model
extract_fully_connected_architecture = _MODULE.extract_fully_connected_architecture
main = _MODULE.main
save_figure = _MODULE.save_figure
visualize_fully_connected_model = _MODULE.visualize_fully_connected_model

__all__ = [
    "LinearLayerSpec",
    "build_demo_model",
    "extract_fully_connected_architecture",
    "main",
    "save_figure",
    "visualize_fully_connected_model",
]
