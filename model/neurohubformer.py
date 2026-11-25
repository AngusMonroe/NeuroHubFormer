"""NeuroHubFormer: subject-specific hubs guiding a token view and a Riemannian view
of the functional connectivity (FC) matrix.

Components (equation numbers refer to the paper):
  * ROITokenizer             ROI tokens h_i = W_in a_i + p_i                    (Eq. 1)
  * HubScorer                interaction gates, density / coverage, hub scores  (Eqs. 2-4)
                             and hub-modulated tokens                           (Eq. 5)
  * HubPriorTransformerLayer pre-LN Transformer layer with a hub attention prior (Eq. 6)
  * HubRiemannBranch         hub-weighted congruence + log-Euclidean embedding  (Eqs. 7-8)
  * NeuroHubFormer           fusion of both views and the classification head   (Eq. 10)
"""
import math
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn


class ROITokenizer(nn.Module):
    """ROI tokens: a shared projection of each connectivity row plus a learnable
    identity embedding per ROI, followed by dropout."""

    def __init__(self, num_nodes, hidden_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.node_identity = nn.Parameter(torch.zeros(num_nodes, hidden_dim))
        self.mlp = nn.Linear(num_nodes, hidden_dim)
        nn.init.kaiming_normal_(self.node_identity)

    def forward(self, connectivity):
        return self.dropout(self.mlp(connectivity) + self.node_identity)


class HubScorer(nn.Module):
    """Differentiable hub scorer.

    Interaction gates g_ij = sigmoid(z_i^T z_j) with z_i = W_MI h_i (g_ii = 0) give,
    per ROI, the density rho_i (mean gate) and the coverage kappa_i (participation
    ratio of the gates over N - 1). The hub logit combines a linear profile score
    u_i = w_u^T h_i + b_u with rho_i and the per-subject z-scored coverage:
        l_i = a_u u_i + a_rho rho_i + a_kappa z(kappa)_i + c,   eta_i = sigmoid(l_i).
    Tokens are then modulated around the identity:
        h'_i = h_i + gamma (2 eta_i - 1) W_mod h_i.

    score_init="random" keeps PyTorch's default initialization of nn.Linear;
    "fixed" starts (a_u, a_rho, a_kappa, c) at (1, 1, 1, 0) in every run.
    """

    def __init__(self, hidden_dim, interaction_dim=64, score_init="random", eps=1e-6):
        super().__init__()
        if score_init not in ("fixed", "random"):
            raise ValueError(f"score_init must be 'fixed' or 'random', got {score_init!r}.")
        self.eps = eps
        self.mi_proj = nn.Linear(hidden_dim, interaction_dim, bias=False)  # W_MI
        self.token_score = nn.Linear(hidden_dim, 1)                         # u_i
        self.score_combine = nn.Linear(3, 1)                                # a_u, a_rho, a_kappa, c
        if score_init == "fixed":
            with torch.no_grad():
                self.score_combine.weight.fill_(1.0)
                self.score_combine.bias.zero_()
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)                 # W_mod
        self.hub_scale = nn.Parameter(torch.tensor(0.1))                    # gamma

    def forward(self, tokens):
        """
        Args:
            tokens: [B, N, d] ROI tokens.
        Returns:
            dict with modulated tokens [B, N, d] and per-ROI [B, N, 1] tensors:
            hub_logits, hub_scores (eta), profile (u), density (rho),
            coverage (raw kappa), coverage_z (z-scored kappa) and entropy
            (normalized gate entropy).
        """
        num_nodes = tokens.size(1)
        projected = self.mi_proj(tokens)
        gates = torch.sigmoid(projected @ projected.transpose(-1, -2))
        non_self = ~torch.eye(num_nodes, dtype=torch.bool, device=tokens.device)
        gates = gates * non_self.float().unsqueeze(0)

        gate_sum = gates.sum(dim=-1, keepdim=True)
        gate_square_sum = (gates * gates).sum(dim=-1, keepdim=True)
        neighbors = max(1, num_nodes - 1)
        density = gate_sum / neighbors
        coverage = (gate_sum * gate_sum) / (gate_square_sum + self.eps) / neighbors

        gate_dist = gates / (gate_sum + self.eps)
        entropy = -(gate_dist * torch.log(gate_dist + self.eps)).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(max(2, num_nodes - 1))

        # Coverage is standardized over the ROIs of each subject; density keeps its
        # absolute scale.
        coverage_mean = coverage.mean(dim=1, keepdim=True)
        coverage_std = torch.sqrt(((coverage - coverage_mean) ** 2).mean(dim=1, keepdim=True) + self.eps)
        coverage_z = (coverage - coverage_mean) / coverage_std

        profile = self.token_score(tokens)
        hub_logits = self.score_combine(torch.cat([profile, density, coverage_z], dim=-1))
        hub_scores = torch.sigmoid(hub_logits)

        modulated = tokens + self.hub_scale * (2.0 * hub_scores - 1.0) * self.value_proj(tokens)
        return {
            "tokens": modulated,
            "hub_logits": hub_logits,
            "hub_scores": hub_scores,
            "profile": profile,
            "density": density,
            "coverage": coverage,
            "coverage_z": coverage_z,
            "entropy": entropy,
        }


class HubPriorTransformerLayer(nn.Module):
    """Pre-LN Transformer layer whose attention logits receive a key-side hub
    prior beta * eta_v, with a learnable per-layer beta initialized to zero."""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.hub_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, hub_vec):
        """x: [B, S, d] tokens; hub_vec: [B, S] hub score per key token."""
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + self.hub_beta * hub_vec[:, None, None, :]
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))

        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        x = x + self.out_dropout(self.out_proj(context))
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Matrix logarithm with a Daleckii-Krein backward pass
# ---------------------------------------------------------------------------

EIGH_BACKENDS = ("cpu", "native")
_EIGH_POOLS = {}


def _eigh_pool(workers):
    """Thread pool whose threads each run single-threaded LAPACK."""
    pool = _EIGH_POOLS.get(workers)
    if pool is None:
        pool = ThreadPoolExecutor(workers, initializer=torch.set_num_threads, initargs=(1,),
                                  thread_name_prefix="eigh")
        _EIGH_POOLS[workers] = pool
    return pool


def batched_eigh(matrices, backend="cpu"):
    """torch.linalg.eigh over a batch.

    backend="cpu" solves the batch on the host, split across torch.get_num_threads()
    threads, and returns the results on the input's device (on CUDA this is much
    faster than the native solver for N > 32, which handles one matrix at a time).
    backend="native" calls torch.linalg.eigh on the input's device.
    """
    if backend == "native":
        return torch.linalg.eigh(matrices)
    if backend != "cpu":
        raise ValueError(f"eigh backend must be one of {EIGH_BACKENDS}, got {backend!r}.")
    device = matrices.device
    host = matrices.detach().to("cpu")
    flat = host.reshape(-1, host.size(-2), host.size(-1))
    workers = max(1, min(flat.size(0), torch.get_num_threads()))
    parts = list(_eigh_pool(workers).map(torch.linalg.eigh, flat.chunk(workers)))
    eigvals = torch.cat([part[0] for part in parts]).reshape(host.shape[:-1])
    eigvecs = torch.cat([part[1] for part in parts]).reshape(host.shape)
    return eigvals.to(device), eigvecs.to(device)


class _SymmetricMatrixLog(torch.autograd.Function):
    """log(S) = U diag(log max(lambda, eps)) U^T for symmetric positive-definite S.

    The backward pass uses the Daleckii-Krein formula: K_ij = (log l_i - log l_j) /
    (l_i - l_j), replaced by the mean derivative when two eigenvalues coincide, so
    the gradient stays finite for repeated eigenvalues. Below eps the derivative is 0.
    """

    @staticmethod
    def forward(ctx, matrices, eps, eigh_backend):
        eigvals, eigvecs = batched_eigh(matrices, eigh_backend)
        clamped = eigvals.clamp_min(eps)
        logs = torch.log(clamped)
        ctx.save_for_backward(eigvals, eigvecs, clamped, logs)
        ctx.eps = eps
        return eigvecs @ torch.diag_embed(logs) @ eigvecs.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        eigvals, eigvecs, clamped, logs = ctx.saved_tensors
        grad = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        grad = eigvecs.transpose(-1, -2) @ grad @ eigvecs
        derivative = torch.where(eigvals > ctx.eps, 1.0 / clamped, torch.zeros_like(clamped))
        value_gap = clamped.unsqueeze(-1) - clamped.unsqueeze(-2)
        log_gap = logs.unsqueeze(-1) - logs.unsqueeze(-2)
        scale = torch.maximum(clamped.unsqueeze(-1), clamped.unsqueeze(-2))
        close = value_gap.abs() <= 1e-9 * scale
        divided = log_gap / torch.where(close, torch.ones_like(value_gap), value_gap)
        mean_derivative = 0.5 * (derivative.unsqueeze(-1) + derivative.unsqueeze(-2))
        kernel = torch.where(close, mean_derivative, divided)
        return eigvecs @ (kernel * grad) @ eigvecs.transpose(-1, -2), None, None


def spd_matrix_log(matrices, eps=1e-4, eigh_backend="cpu"):
    """Matrix logarithm of a batch of SPD matrices, computed in float64."""
    return _SymmetricMatrixLog.apply(matrices.double(), eps, eigh_backend).to(matrices.dtype)


class HubRiemannBranch(nn.Module):
    """Hub-weighted log-Euclidean embedding of the FC matrix.

    Per subject: C_alpha = (1 - alpha) A + alpha I; C_l = D C_alpha D with
    D = diag(sqrt(w)), w = N eta / sum(eta); t = vechs(log C_l) (upper triangle,
    off-diagonals scaled by sqrt(2)); z_R = GELU(W_R Dropout(LayerNorm(t))).
    """

    def __init__(self, num_nodes, out_dim, shrinkage=0.1, dropout=0.1, eps=1e-4, eigh_backend="cpu"):
        super().__init__()
        if not 0.0 < shrinkage < 1.0:
            raise ValueError("shrinkage must lie in (0, 1).")
        if eigh_backend not in EIGH_BACKENDS:
            raise ValueError(f"eigh_backend must be one of {EIGH_BACKENDS}, got {eigh_backend!r}.")
        self.shrinkage = float(shrinkage)
        self.eps = eps
        self.eigh_backend = eigh_backend
        rows, cols = torch.triu_indices(num_nodes, num_nodes)
        self.register_buffer("tri_rows", rows, persistent=False)
        self.register_buffer("tri_cols", cols, persistent=False)
        scale = torch.where(rows == cols, torch.ones(rows.numel()), torch.full((rows.numel(),), math.sqrt(2.0)))
        self.register_buffer("tri_scale", scale, persistent=False)
        feature_dim = num_nodes * (num_nodes + 1) // 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, connectivity, hub_eta):
        """
        Args:
            connectivity: [B, N, N] FC matrix with a zero diagonal.
            hub_eta: [B, N, 1] hub scores eta = sigmoid(hub logits).
        Returns:
            [B, out_dim] features of the geometric view.
        """
        num_nodes = connectivity.size(-1)
        eye = torch.eye(num_nodes, device=connectivity.device, dtype=connectivity.dtype)
        corr = connectivity + eye
        corr = 0.5 * (corr + corr.transpose(-1, -2))
        corr = (1.0 - self.shrinkage) * corr + self.shrinkage * eye

        eta = hub_eta.squeeze(-1)
        eta_sum = eta.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(eta.dtype).tiny)
        root = torch.sqrt(num_nodes * eta / eta_sum)
        corr = root.unsqueeze(-1) * corr * root.unsqueeze(-2)
        corr = 0.5 * (corr + corr.transpose(-1, -2))

        log_map = spd_matrix_log(corr, eps=self.eps, eigh_backend=self.eigh_backend)
        tangent = log_map[:, self.tri_rows, self.tri_cols] * self.tri_scale
        return self.mlp(tangent)


class NeuroHubFormer(nn.Module):
    """Hub scorer + hub-guided Transformer (token view) + hub-weighted Riemannian
    embedding (geometric view), fused by concatenation before a two-layer head.

    After each forward pass, `hub_entropy` and `hub_coverage` ([B, N, 1]) hold the
    quantities of the gate-entropy and coverage losses.
    """

    def __init__(
        self,
        num_nodes,
        num_classes,
        hidden_dim=128,
        num_layers=1,
        num_heads=8,
        ff_dim=128,
        dropout=0.1,
        interaction_dim=64,
        shrinkage=0.1,
        eigh_backend="cpu",
        hub_score_init="random",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.id_enc = ROITokenizer(num_nodes, hidden_dim, dropout)
        self.hub_scorer = HubScorer(hidden_dim, interaction_dim=interaction_dim, score_init=hub_score_init)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = nn.ModuleList(
            [HubPriorTransformerLayer(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.riemann_branch = HubRiemannBranch(
            num_nodes, hidden_dim, shrinkage=shrinkage, dropout=dropout, eigh_backend=eigh_backend
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        self.hub_entropy = None
        self.hub_coverage = None

    def forward(self, x, return_hub_scores=False):
        """
        Args:
            x: [B, N, N] Pearson correlation matrices.
            return_hub_scores: also return the per-ROI hub quantities of HubScorer.
        Returns:
            logits [B, num_classes] (and the hub dict if return_hub_scores).
        """
        if x.dim() != 3 or x.size(1) != self.num_nodes or x.size(2) != self.num_nodes:
            raise ValueError(f"Expected input of shape (B, {self.num_nodes}, {self.num_nodes}), got {tuple(x.shape)}.")

        # Self-connections are removed from all inputs.
        non_self = ~torch.eye(self.num_nodes, device=x.device, dtype=torch.bool)
        connectivity = x * non_self.to(dtype=x.dtype).unsqueeze(0)

        hub = self.hub_scorer(self.id_enc(connectivity))
        self.hub_entropy = hub["entropy"]
        self.hub_coverage = hub["coverage"]
        hub_scores = hub["hub_scores"]

        # Token view: CLS + modulated ROI tokens; CLS gets a neutral hub prior of 0.
        batch_size = x.size(0)
        tokens = torch.cat([self.cls_token.expand(batch_size, -1, -1), hub["tokens"]], dim=1)
        hub_vec = hub_scores.squeeze(-1)
        hub_vec = torch.cat([hub_vec.new_zeros(batch_size, 1), hub_vec], dim=1)
        for layer in self.encoder:
            tokens = layer(tokens, hub_vec)
        token_view = self.norm(tokens)[:, 0]

        # Geometric view.
        geometric_view = self.riemann_branch(connectivity, hub_scores)

        logits = self.head(torch.cat([token_view, geometric_view], dim=-1))
        if return_hub_scores:
            return logits, {k: v for k, v in hub.items() if k != "tokens"}
        return logits
