"""Core utilities for the Modern Edge Detection Toolkit.

This module provides helper functions to load configuration files,
list input images and run the six edge detection methods defined in
the project README. The actual network implementations are expected
as submodules located inside ``modern_edge_toolkit``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
import yaml


# Supported methods and image extensions
SUPPORTED_METHODS: Sequence[str] = (
    "BDCN",
    "RCF",
    "HED",
    "DexiNed",
    "CASENet",
    "Structured",
)

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".jp2",
}


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML configuration.

    Parameters
    ----------
    path:
        Optional path to the YAML file. If omitted, ``config_modern.yaml``
        in the package root is used.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "config_modern.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_input_images(directory: str | os.PathLike) -> List[Path]:
    """Return sorted list of image files in ``directory``."""
    dir_path = Path(directory)
    images = [p for p in dir_path.iterdir() if p.suffix.lower() in VALID_EXTS]
    images.sort()
    return images


def _device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_bdcn(cfg: dict, device: torch.device):
    repo = Path(__file__).resolve().parent / "bdcn_repo"
    sys.path.insert(0, str(repo))
    weight = repo / "pretrained" / "bdcn_pretrained.pth"
    if not weight.exists():
        raise FileNotFoundError(weight)
    try:
        from models.bdcn import BDCN  # type: ignore
    except Exception as exc:  # pragma: no cover - import paths may vary
        raise ImportError("BDCN repository missing or invalid") from exc
    model = BDCN()
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_rcf(cfg: dict, device: torch.device):
    repo = Path(__file__).resolve().parent / "rcf_repo"
    sys.path.insert(0, str(repo))
    weight = repo / "model" / "RCF.pth"
    if not weight.exists():
        raise FileNotFoundError(weight)
    try:
        from model import RCF  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RCF repository missing or invalid") from exc
    model = RCF()
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_dexined(cfg: dict, device: torch.device):
    repo = Path(__file__).resolve().parent / "dexined_repo"
    sys.path.insert(0, str(repo))
    weight = repo / "weights" / "dexined.pth"
    if not weight.exists():
        raise FileNotFoundError(weight)
    try:
        from DexiNed.model import DexiNed  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("DexiNed repository missing or invalid") from exc
    model = DexiNed()
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_casenet(cfg: dict, device: torch.device):
    repo = Path(__file__).resolve().parent / "casenet_repo"
    sys.path.insert(0, str(repo))
    weight = repo / "model" / "casenet.pth"
    if not weight.exists():
        raise FileNotFoundError(weight)
    try:
        from network import CASENet  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("CASENet repository missing or invalid") from exc
    model = CASENet()
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_hed(cfg: dict):
    repo = Path(__file__).resolve().parent / "hed_repo"
    proto = repo / "deploy.prototxt"
    model = repo / "hed_pretrained_bsds.caffemodel"
    if not proto.exists() or not model.exists():
        raise FileNotFoundError("HED prototxt or caffemodel missing")
    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    return net


def _load_structured(cfg: dict):
    model_path = Path(__file__).resolve().parent / "models" / "structured" / "model.yml.gz"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return cv2.ximgproc.createStructuredEdgeDetection(str(model_path))


LOADERS = {
    "BDCN": _load_bdcn,
    "RCF": _load_rcf,
    "DexiNed": _load_dexined,
    "CASENet": _load_casenet,
    "HED": _load_hed,
    "Structured": _load_structured,
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_torch(model: torch.nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()
    return pred


def _infer_hed(net, img: np.ndarray) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=img.shape[:2][::-1],
                                mean=(104.00699, 116.66877, 122.67892), swapRB=False)
    net.setInput(blob)
    out = net.forward()
    return out[0, 0]


def _infer_structured(detector, img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detector.detectEdges(np.float32(rgb) / 255.0)


def detect_image(img_path: Path, method: str, model, device: torch.device) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    if method == "HED":
        edge = _infer_hed(model, img)
    elif method == "Structured":
        edge = _infer_structured(model, img)
    else:
        edge = _infer_torch(model, img, device)
    edge = (edge * 255.0).clip(0, 255).astype("uint8")
    return edge


def save_edge(edge: np.ndarray, out_dir: Path, method: str, name: str) -> Path:
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    out_path = method_dir / f"{name}.png"
    cv2.imwrite(str(out_path), edge)
    return out_path


def clear_gpu():
    if torch.cuda.is_available():  # pragma: no cover - only on GPU systems
        torch.cuda.empty_cache()


def process_images(
    images: Iterable[Path],
    methods: Sequence[str],
    cfg: dict,
    progress_callback=None,
) -> List[Path]:
    device = _device(cfg.get("use_gpu", False))
    results: List[Path] = []
    total = len(list(images)) * len(methods)
    count = 0
    images = list(images)
    for method in methods:
        loader = LOADERS.get(method)
        if loader is None:
            raise ValueError(f"Unknown method {method}")
        model = loader(cfg, device)
        for img in images:
            edge = detect_image(img, method, model, device)
            out = save_edge(edge, Path(cfg["output_dir"]), method, img.stem)
            results.append(out)
            count += 1
            if progress_callback is not None:
                progress_callback(count / total)
        clear_gpu()
    if progress_callback is not None:
        progress_callback(1.0)
    return results
