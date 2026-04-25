from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils.cache_utils import configure_model_cache

configure_model_cache()


def _resolve_dinov2_model_name(model_name: Optional[str]) -> str:
    name = "" if model_name is None else str(model_name).strip()
    if len(name) == 0 or name == "facebookresearch/dinov2":
        return "dinov2_vitb14"
    if ":" in name:
        repo_name, hub_name = name.split(":", 1)
        if repo_name.strip() == "facebookresearch/dinov2" and len(hub_name.strip()) > 0:
            return hub_name.strip()
    return name


def _scaled_dinov2_size(size: int, base_patch_size: int = 16, dino_patch_size: int = 14) -> int:
    scaled = int(float(dino_patch_size) * float(int(size)) / max(1, int(base_patch_size)))
    return max(int(dino_patch_size), scaled)


def _compute_dinov2_resize_hw(
    h: int,
    w: int,
    base_patch_size: int = 16,
    dino_patch_size: int = 14,
) -> Tuple[int, int]:
    target_h = _scaled_dinov2_size(h, base_patch_size=base_patch_size, dino_patch_size=dino_patch_size)
    target_w = _scaled_dinov2_size(w, base_patch_size=base_patch_size, dino_patch_size=dino_patch_size)
    return target_h, target_w


def _parse_resolution_list(
    values: Optional[Union[str, int, List[int], Tuple[int, ...]]],
    default: Optional[List[int]] = None,
) -> List[int]:
    if values is None:
        src = [] if default is None else list(default)
    elif isinstance(values, str):
        token = values.strip().lower()
        if token in {"", "none", "off"}:
            src = []
        elif token == "default":
            src = [] if default is None else list(default)
        elif token in {"primary", "single"}:
            src = [] if default is None or len(default) == 0 else [default[0]]
        else:
            src = [v.strip() for v in values.split(",")]
    elif isinstance(values, int):
        src = [values]
    else:
        src = list(values)

    out: List[int] = []
    for item in src:
        if item is None:
            continue
        if isinstance(item, str):
            item = item.strip()
            if len(item) == 0:
                continue
        res = int(item)
        if res <= 0:
            raise ValueError(f"Resolution must be positive, got {res}")
        if res not in out:
            out.append(res)
    return out


def _sample_shared_patch_idx(
    patch_sets: List[torch.Tensor],
    token_subsample: int,
) -> Optional[torch.Tensor]:
    if int(token_subsample) <= 0:
        return None
    min_tokens = min(int(p.shape[1]) for p in patch_sets)
    if int(token_subsample) >= min_tokens:
        return None
    return torch.randperm(min_tokens, device=patch_sets[0].device)[: int(token_subsample)]


def _self_sim_matrix_from_patches(
    patches: torch.Tensor,
    token_subsample: int = 0,
    idx: Optional[torch.Tensor] = None,
    l2norm: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _, n_tokens, _ = patches.shape
    if idx is None and token_subsample and token_subsample < n_tokens:
        idx = torch.randperm(n_tokens, device=patches.device)[:token_subsample]
    if idx is not None:
        patches = patches[:, idx, :]

    p = patches.float()
    if l2norm:
        p = F.normalize(p, dim=-1, eps=eps)

    sim = torch.bmm(p, p.transpose(1, 2))
    return sim, idx


def _build_second_order_gram(
    gram: torch.Tensor,
    remove_diag: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    g_rel = gram
    if remove_diag:
        n = int(gram.shape[-1])
        eye = torch.eye(n, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        g_rel = gram * (1.0 - eye)

    r = F.normalize(g_rel, dim=-1, eps=float(eps))
    return torch.bmm(r, r.transpose(1, 2))


def _kl_ref_from_logits(
    logits_ref: torch.Tensor,
    logits_pred: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    tau = max(float(tau), float(eps))
    log_ref = F.log_softmax(logits_ref / tau, dim=-1)
    log_pred = F.log_softmax(logits_pred / tau, dim=-1)
    p_ref = log_ref.exp()
    return (p_ref * (log_ref - log_pred)).sum(dim=-1).mean()


class SecondOrderDinoGramLoss(nn.Module):
    """
    Second-order Gram KL loss from DINOv2 patch-token self-similarity.

    Inputs are expected to be BCHW RGB tensors in [-1, 1]. Gradients flow through
    pred; target is encoded under no_grad and used as the reference distribution.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        resize: int = 256,
        resolutions: Optional[Union[str, int, List[int], Tuple[int, ...]]] = None,
        token_subsample: int = 0,
        tau: float = 0.1,
        eps: float = 1e-6,
        remove_diag: bool = False,
        l2norm: bool = True,
        base_patch_size: int = 16,
    ):
        super().__init__()
        self.model_name = _resolve_dinov2_model_name(model_name)
        self.resize = int(resize)
        default_resolutions = [self.resize] if self.resize > 0 else []
        self.resolutions = _parse_resolution_list(resolutions, default=default_resolutions)
        self.token_subsample = int(token_subsample)
        self.tau = float(tau)
        self.eps = float(eps)
        self.remove_diag = bool(remove_diag)
        self.l2norm = bool(l2norm)
        self.base_patch_size = int(base_patch_size)
        self.dino_patch_size = 14

        self.net = torch.hub.load("facebookresearch/dinov2", self.model_name)
        if hasattr(self.net, "head"):
            self.net.head = nn.Identity()
        self.net.eval()
        self.net.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _preprocess(self, x: torch.Tensor, resize_to: Optional[int]) -> torch.Tensor:
        x01 = (x.float().clamp(-1, 1) + 1.0) * 0.5

        if resize_to is not None and resize_to > 0 and (x01.shape[-2] != resize_to or x01.shape[-1] != resize_to):
            x01 = F.interpolate(
                x01,
                size=(int(resize_to), int(resize_to)),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        target_h, target_w = _compute_dinov2_resize_hw(
            int(x01.shape[-2]),
            int(x01.shape[-1]),
            base_patch_size=self.base_patch_size,
            dino_patch_size=self.dino_patch_size,
        )
        if x01.shape[-2] != target_h or x01.shape[-1] != target_w:
            x01 = F.interpolate(
                x01,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        return (x01 - self.mean) / self.std

    def forward_tokens(self, x: torch.Tensor, resize_to: Optional[int] = None) -> torch.Tensor:
        out = self.net.forward_features(self._preprocess(x, resize_to=resize_to))
        if not isinstance(out, dict):
            raise RuntimeError(f"DINOv2 forward_features must return a dict, got {type(out)}")
        patches = out.get("x_norm_patchtokens", None)
        if patches is None:
            raise RuntimeError("DINOv2 forward_features output is missing 'x_norm_patchtokens'.")
        return patches

    def _pair_loss_from_patches(
        self,
        pred_patches: torch.Tensor,
        target_patches: torch.Tensor,
    ) -> torch.Tensor:
        idx = _sample_shared_patch_idx([pred_patches, target_patches], self.token_subsample)
        pred_gram, _ = _self_sim_matrix_from_patches(
            pred_patches,
            token_subsample=0,
            idx=idx,
            l2norm=self.l2norm,
            eps=self.eps,
        )
        target_gram, _ = _self_sim_matrix_from_patches(
            target_patches,
            token_subsample=0,
            idx=idx,
            l2norm=self.l2norm,
            eps=self.eps,
        )

        pred_second = _build_second_order_gram(pred_gram, remove_diag=self.remove_diag, eps=self.eps)
        target_second = _build_second_order_gram(target_gram, remove_diag=self.remove_diag, eps=self.eps)
        return _kl_ref_from_logits(
            logits_ref=target_second,
            logits_pred=pred_second,
            tau=self.tau,
            eps=self.eps,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        resolutions: List[Optional[int]] = list(self.resolutions)
        if len(resolutions) == 0:
            resolutions = [None]

        total = pred.new_tensor(0.0, dtype=torch.float32)
        for res in resolutions:
            pred_patches = self.forward_tokens(pred, resize_to=res)
            with torch.no_grad():
                target_patches = self.forward_tokens(target, resize_to=res)
            total = total + self._pair_loss_from_patches(pred_patches, target_patches)
        return total / float(len(resolutions))
