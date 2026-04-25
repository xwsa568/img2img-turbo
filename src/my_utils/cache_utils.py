import os
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_ENV = "IMG2IMG_TURBO_CKPT_DIR"


def get_model_cache_root(cache_dir: Optional[str] = None) -> Path:
    raw_cache_dir = cache_dir or os.environ.get(MODEL_CACHE_ENV) or "ckpts"
    root = Path(raw_cache_dir).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def configure_model_cache(cache_dir: Optional[str] = None) -> Dict[str, Path]:
    root = get_model_cache_root(cache_dir)
    paths = {
        "root": root,
        "huggingface": root / "huggingface",
        "hf_hub": root / "huggingface" / "hub",
        "transformers": root / "huggingface" / "transformers",
        "diffusers": root / "huggingface" / "diffusers",
        "torch": root / "torch",
        "torch_hub": root / "torch" / "hub",
        "clip": root / "clip",
        "xdg": root / "xdg",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ[MODEL_CACHE_ENV] = paths["root"].as_posix()
    os.environ["HF_HOME"] = paths["huggingface"].as_posix()
    os.environ["HF_HUB_CACHE"] = paths["hf_hub"].as_posix()
    os.environ["HUGGINGFACE_HUB_CACHE"] = paths["hf_hub"].as_posix()
    os.environ["TRANSFORMERS_CACHE"] = paths["transformers"].as_posix()
    os.environ["DIFFUSERS_CACHE"] = paths["diffusers"].as_posix()
    os.environ["TORCH_HOME"] = paths["torch"].as_posix()
    os.environ["XDG_CACHE_HOME"] = paths["xdg"].as_posix()
    os.environ["CLIP_CACHE_DIR"] = paths["clip"].as_posix()

    try:
        import torch

        torch.hub.set_dir(paths["torch_hub"].as_posix())
    except Exception:
        pass

    return paths


def get_hf_cache_dir(cache_dir: Optional[str] = None) -> str:
    return configure_model_cache(cache_dir)["hf_hub"].as_posix()


def get_clip_cache_dir(cache_dir: Optional[str] = None) -> str:
    return configure_model_cache(cache_dir)["clip"].as_posix()
