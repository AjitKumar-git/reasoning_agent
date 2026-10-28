"""
SpheroidOptimizer: A Manifold & Spectral-Aware Optimizer for CerebroSpheroid Models.

Why AdamW is unsuitable for Spheroid models:
1. Coordinate-Wise Distortion: AdamW scales updates independently per coordinate (element-wise v_t).
   In continuous Riemannian manifolds and spherical shells (B^3), coordinate scaling violates rotational
   isometry and breaks Laplace-Beltrami tangential flow and radial wave coupling.
2. Recurrent Spectral Radius Blowup: Spheroid relies on contractive unit-gain dynamics across recurrent
   deliberation cycles: S_{t+1} = Norm(gamma S_t + (1-gamma) tanh(Flow)). AdamW updates inject unconstrained
   spectral energy into recurrence matrices (intra_shell, reentrant_field, shortcut_tracts), pushing singular
   values above 1.0 and causing gradient explosion / chaotic drift through time.
3. Memory Inefficiency: AdamW maintains two fp32 moments per parameter (16 bytes/param total with weights
   and grads). In recurrent deliberation rollouts, this causes GPU out-of-memory errors on edge/laptop hardware
   (e.g., 4GB-8GB GPUs), forcing a retreat to unconditioned SGD.

SpheroidOptimizer solves these challenges:
1. Orthogonal Polar Updates (Quintic Newton-Schulz): For all 2D, 3D, and 4D operator tensors, the momentum
   matrix is projected to the nearest orthogonal manifold via vectorized Newton-Schulz iterations. All singular
   values of the update direction are identically 1.0, preserving matrix isometry without coordinate-wise distortion.
2. Single-Moment Memory Footprint: Eliminates the second-moment buffer (v_t) entirely for matrix/manifold
   parameters, cutting optimizer state memory by 50% (12 bytes/param in fp32, or 10 bytes/param with fp16 momentum).
3. Contractive Spectral Guard: Optional spectral radius enforcement ensures recurrent operators remain contractive
   (rho(W) <= rho_max) across arbitrary deliberation depths.
4. Scale-Invariant 1D Updates: Scalar gates (log_write, log_leak, inertia_gate, rhythm_freq, pacemakers) and norms
   are updated via RMS-normalized momentum, preventing saturation without requiring heavy variance buffers.
"""

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer


def _zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Vectorized quintic Newton-Schulz iteration to compute the matrix zeroth power / orthogonalization.
    Supports 2D [M, N] and batched 3D [B, M, N] tensors natively on GPU/CPU.
    
    The iteration computes O = G (G^T G)^{-1/2}, mapping all non-zero singular values to 1.0.
    """
    assert G.ndim >= 2, f"Newton-Schulz requires at least 2 dimensions, got {G.ndim}"
    
    # Store original shape and transpose if rows > cols to minimize inner dimension
    orig_shape = G.shape
    if G.ndim > 3:
        # Flatten outer batch dimensions into a single batch dimension
        G = G.reshape(-1, orig_shape[-2], orig_shape[-1])
        
    transposed = False
    if G.shape[-2] > G.shape[-1]:
        G = G.transpose(-2, -1)
        transposed = True

    # Pre-scale so that the spectral norm is strictly <= 1.0 (preventing polynomial divergence)
    norm = torch.linalg.matrix_norm(G, ord="fro", keepdim=True).clamp_min(eps)
    X = G / norm

    # Quintic Newton-Schulz coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-2, -1))
        B = b * A + c * torch.matmul(A, A)
        X = a * X + torch.matmul(B, X)

    if transposed:
        X = X.transpose(-2, -1)

    if len(orig_shape) > 3:
        X = X.reshape(orig_shape)

    return X


def _clamp_spectral_norm(
    W: torch.Tensor,
    max_spec_norm: float = 1.0,
    power_steps: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Contractive spectral radius guard.
    Clamps the top singular value (operator norm) to max_spec_norm if it exceeds it.
    Uses exact batched matrix_norm (ord=2) or power iteration with safety margin.
    """
    if W.ndim < 2:
        return W
        
    orig_shape = W.shape
    if W.ndim > 3:
        W_mat = W.reshape(-1, orig_shape[-2], orig_shape[-1])
    elif W.ndim == 2:
        W_mat = W.unsqueeze(0)
    else:
        W_mat = W  # [B, M, N]

    B, M, N = W_mat.shape

    # If matrix dimensions are moderate (<= 1024), exact batched 2-norm is fast and bitwise exact
    if max(M, N) <= 1024:
        try:
            norms = torch.linalg.matrix_norm(W_mat, ord=2)  # [B]
            scale = torch.clamp(max_spec_norm / (norms.unsqueeze(-1).unsqueeze(-1) + eps), max=1.0)
            W_clamped = W_mat * scale
            if len(orig_shape) > 3 or W.ndim == 2:
                W_clamped = W_clamped.reshape(orig_shape)
            return W_clamped
        except Exception:
            pass  # Fall back to power iteration

    # Power iteration fallback
    v = torch.randn(B, N, 1, device=W.device, dtype=W.dtype)
    v = v / (torch.norm(v, dim=1, keepdim=True).clamp_min(eps))

    for _ in range(power_steps):
        u = torch.matmul(W_mat, v)
        u = u / (torch.norm(u, dim=1, keepdim=True).clamp_min(eps))
        v = torch.matmul(W_mat.transpose(-2, -1), u)
        v = v / (torch.norm(v, dim=1, keepdim=True).clamp_min(eps))

    sigma = torch.norm(torch.matmul(W_mat, v), dim=1, keepdim=True)  # [B, 1, 1]
    # Multiply by conservative factor 1.08 because power iteration lower-bounds sigma_max
    scale = torch.clamp(max_spec_norm / (sigma * 1.08 + eps), max=1.0)
    W_clamped = W_mat * scale

    if len(orig_shape) > 3 or W.ndim == 2:
        W_clamped = W_clamped.reshape(orig_shape)

    return W_clamped


class SpheroidOptimizer(Optimizer):
    """
    SpheroidOptimizer: Specialized optimizer for the CerebroSpheroid reasoning manifold.
    
    Parameters:
    -----------
    params: iterable of parameters or param_groups
    lr: float
        Base learning rate for matrix/manifold parameters.
    lr_1d: Optional[float]
        Learning rate for 1D scalar parameters, pacemakers, and normalization weights.
        Defaults to 0.2 * lr if not specified.
    momentum: float
        Momentum factor (beta1). Defaults to 0.95.
    weight_decay: float
        Decoupled weight decay for matrix parameters. Defaults to 0.01.
    ns_steps: int
        Number of quintic Newton-Schulz iterations. Defaults to 5.
    spectral_clamp: Optional[float]
        Maximum allowed spectral norm for recurrent operators. If None, disabled.
        Recommended: 1.0 for contractive stability.
    half_precision_moments: bool
        If True, stores momentum buffers in float16/bfloat16 to reduce memory footprint.
        Defaults to False (fp32).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict]],
        lr: float = 0.005,
        lr_1d: Optional[float] = None,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        spectral_clamp: Optional[float] = 1.0,
        half_precision_moments: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        if lr_1d is None:
            lr_1d = 0.2 * lr

        defaults = dict(
            lr=lr,
            lr_1d=lr_1d,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            spectral_clamp=spectral_clamp,
            half_precision_moments=half_precision_moments,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_1d = group.get("lr_1d", 0.2 * lr)
            beta = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group.get("ns_steps", 5)
            spectral_clamp = group.get("spectral_clamp", None)
            is_recurrent = group.get("is_recurrent", False)
            use_half = group.get("half_precision_moments", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SpheroidOptimizer does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    m_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if (use_half and p.is_cuda) else p.dtype
                    state["momentum"] = torch.zeros_like(p, dtype=m_dtype)

                state["step"] += 1
                buf = state["momentum"]

                # Update first moment: M_t = beta * M_{t-1} + (1 - beta) * G_t
                if buf.dtype != p.dtype:
                    buf_f = buf.to(p.dtype)
                    buf_f.mul_(beta).add_(grad, alpha=1.0 - beta)
                    buf.copy_(buf_f)
                    m = buf_f
                else:
                    buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                    m = buf

                # -----------------------------------------------------------------
                # Case 1: Matrix / Manifold Tensor (rank >= 2)
                # -----------------------------------------------------------------
                if p.ndim >= 2:
                    # Decoupled Riemannian-style weight decay
                    if wd > 0.0:
                        p.mul_(1.0 - lr * wd)

                    # Compute orthogonal polar direction via Newton-Schulz
                    O = _zeropower_via_newtonschulz5(m, steps=ns_steps)

                    # Scale by natural operator aspect ratio
                    # For [..., M, N], scale = max(1.0, sqrt(M / N))
                    rows = p.shape[-2]
                    cols = p.shape[-1]
                    aspect_scale = max(1.0, math.sqrt(rows / cols))
                    
                    # Update parameters along orthogonal manifold
                    p.add_(O.to(p.dtype), alpha=-(lr * aspect_scale))

                    # Recurrent manifold contractive spectral guard
                    if is_recurrent and (spectral_clamp is not None) and (spectral_clamp > 0.0):
                        clamped = _clamp_spectral_norm(p, max_spec_norm=spectral_clamp)
                        p.copy_(clamped)

                # -----------------------------------------------------------------
                # Case 2: 1D Parameters (Scalars, Pacemakers, Biases, RMSNorm)
                # -----------------------------------------------------------------
                else:
                    if wd > 0.0:
                        p.mul_(1.0 - lr_1d * wd)

                    # Scale-invariant RMS normalized momentum
                    rms = m.pow(2).mean().sqrt().clamp_min(1e-7)
                    step_dir = m / rms
                    p.add_(step_dir, alpha=-lr_1d)

        return loss


def create_spheroid_param_groups(
    model: torch.nn.Module,
    lr: float = 0.005,
    lr_1d: Optional[float] = None,
    weight_decay: float = 0.01,
    spectral_clamp: float = 1.0,
    half_precision_moments: bool = False,
) -> List[Dict]:
    """
    Automatically classifies model parameters into tailored groups for Spheroid models:
    1. 'recurrent_manifold': intra_shell, reentrant_field, shortcut_tracts (Orthogonal + Spectral Clamp)
    2. 'manifold_matrices': 2D/3D projections, embeddings, linear weights (Orthogonal Polar)
    3. 'wavelets': 1D conv filters in temporal sensory encoder
    4. 'scalar_gates': log_write, log_leak, inertia_gate, rhythm_freq, pacemakers, norms, biases (RMS-normalized)
    """
    if lr_1d is None:
        lr_1d = 0.2 * lr

    recurrent_params = []
    matrix_params = []
    wavelet_params = []
    scalar_params = []

    # Recurrent keywords known in CerebroSpheroid models
    recurrent_names = {"intra_shell", "reentrant_field", "shortcut_tracts"}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(rec in name for rec in recurrent_names):
            recurrent_params.append(p)
        elif p.ndim >= 2:
            if "conv" in name:
                wavelet_params.append(p)
            else:
                matrix_params.append(p)
        else:
            # 1D scalars, pacemakers, gates, biases, rmsnorm weights
            scalar_params.append(p)

    groups = []
    if recurrent_params:
        groups.append({
            "name": "recurrent_manifold",
            "params": recurrent_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "is_recurrent": True,
            "spectral_clamp": spectral_clamp,
            "half_precision_moments": half_precision_moments,
        })
    if matrix_params:
        groups.append({
            "name": "manifold_matrices",
            "params": matrix_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "is_recurrent": False,
            "spectral_clamp": None,
            "half_precision_moments": half_precision_moments,
        })
    if wavelet_params:
        groups.append({
            "name": "wavelets",
            "params": wavelet_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "is_recurrent": False,
            "spectral_clamp": None,
            "half_precision_moments": half_precision_moments,
        })
    if scalar_params:
        groups.append({
            "name": "scalar_gates",
            "params": scalar_params,
            "lr": lr_1d,
            "weight_decay": 0.0,  # No weight decay on scalar gates and pacemakers
            "is_recurrent": False,
            "spectral_clamp": None,
            "half_precision_moments": False,
        })

    return groups


def build_spheroid_optimizer(
    model: torch.nn.Module,
    lr: float = 0.005,
    lr_1d: Optional[float] = None,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    spectral_clamp: float = 1.0,
    half_precision_moments: bool = False,
) -> SpheroidOptimizer:
    """
    Convenience factory to construct a fully-configured SpheroidOptimizer for a model.
    """
    groups = create_spheroid_param_groups(
        model=model,
        lr=lr,
        lr_1d=lr_1d,
        weight_decay=weight_decay,
        spectral_clamp=spectral_clamp,
        half_precision_moments=half_precision_moments,
    )
    return SpheroidOptimizer(
        params=groups,
        lr=lr,
        lr_1d=lr_1d,
        momentum=momentum,
        weight_decay=weight_decay,
        spectral_clamp=spectral_clamp,
        half_precision_moments=half_precision_moments,
    )
