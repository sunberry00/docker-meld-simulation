import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================= LoRALinear

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear: shared base W0 + low-rank site correction.

    Effective transform: y = x W0^T + scaling * (x A^T) B^T, with scaling = alpha / r
    (Klein et al., Eq. 4-5) [cite: 102, 117]. Initialization follows Klein et al.: 
    A ~ N(0, 2/r) and B ~ N(0, 0.01^2)[cite: 114]. This deliberately differs from standard 
    LoRA (B = 0), which they report harmed training due to vanishing gradients through A[cite: 115].
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.linear = linear
        in_f, out_f = linear.in_features, linear.out_features
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * (2.0 / rank) ** 0.5)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scaling * (x @ self.lora_A.T) @ self.lora_B.T


# ============================================================= MLP backbone

class MLP(nn.Module):
    """Backbone MLP. Shared across sites; used for FL training and deployed inference."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 32]):
        super().__init__()
        dims = [input_dim] + hidden_dims + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ============================================================= FT-Transformer backbone

class FeatureTokenizer(nn.Module):
    """Gorishniy et al. (2021): each scalar feature x_i -> d_model-dimensional token."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_model))
        self.b = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F)
        return x.unsqueeze(-1) * self.W + self.b           # (B, F, D)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with explicit Q, K, V, out projections.

    Projections are separate nn.Linear modules so LoRA can be applied
    to Q, K, V independently, matching Klein et al. (2026) [cite: 167].
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        B, T, D = x.shape
        h, dh = self.n_heads, self.d_head

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, h, dh).transpose(1, 2)  # (B, h, T, dh)

        q, k, v = _split(self.q(x)), _split(self.k(x)), _split(self.v(x))
        attn = F.softmax(q @ k.transpose(-2, -1) / dh ** 0.5, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    """Pre-norm block: LayerNorm -> Attention -> Dropout -> residual, LayerNorm -> FFN -> Dropout -> residual.

    Nutzt GELU und Dropout (0.1) analog zum Patched Brain Transformer Setup.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),                  # ReLU durch GELU ersetzt 
            nn.Dropout(dropout),        # Dropout nach Aktivierung 
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),        # Dropout am Ende des FFN Blocks 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual-Verbindung nach Attention + Dropout
        x = x + self.attn_dropout(self.attn(self.norm1(x)))
        # Residual-Verbindung nach FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al. 2021) for binary tabular classification."""

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ffn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F)
        tokens = self.tokenizer(x)                          # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)     # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, F+1, D)
        for block in self.blocks:
            tokens = block(tokens)
        return torch.sigmoid(self.head(self.norm(tokens[:, 0])))  # (B, 1)


class SharedTokenizer(nn.Module):
    """Simple/vanilla tokenizer: ONE shared projection for every scalar feature,
    plus a learned per-feature positional embedding.

    This is the deliberate contrast to FeatureTokenizer (FT-Transformer), which
    learns a SEPARATE W_i per feature. Same attention stack, different tokenization
    -> the comparison isolates the effect of feature-wise tokenization.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)                     # shared across features
        self.pos = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, F)
        return self.proj(x.unsqueeze(-1)) + self.pos         # (B, F, D)


class SimpleTransformer(nn.Module):
    """'Einfacher Transformer': shared scalar embedding + positional embedding.

    Identical block stack to FTTransformer; only the tokenization differs.
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ffn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = SharedTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return torch.sigmoid(self.head(self.norm(tokens[:, 0])))


class TabTransformer(nn.Module):
    """TabTransformer (Huang et al. 2020).

    Key architectural idea, and the reason it differs from FT-Transformer:
    ONLY the categorical features are contextualised by the transformer. The
    continuous features bypass attention entirely, are layer-normed, and are
    concatenated with the transformer output just before the MLP head.

    Our design matrix is already one-hot encoded, so each categorical VARIABLE is
    a contiguous group of dummy columns. A linear map over one group is exactly an
    embedding lookup (one_hot @ W = the selected row of W), so `cat_groups` gives
    a faithful per-variable embedding without re-encoding the data.

    cat_groups : list of column-index lists, one per categorical variable
    num_idx    : column indices of the continuous / binary features
    """

    def __init__(self, n_features: int, cat_groups: list[list[int]], num_idx: list[int],
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ffn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.cat_groups = [list(g) for g in cat_groups]
        self.num_idx = list(num_idx)

        # One embedding matrix per categorical variable (implemented as Linear
        # without bias over that variable's one-hot slice).
        self.cat_embed = nn.ModuleList(
            [nn.Linear(len(g), d_model, bias=False) for g in self.cat_groups]
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.num_norm = nn.LayerNorm(len(self.num_idx)) if self.num_idx else None

        # Head sees: flattened contextualised categorical tokens + raw continuous.
        head_in = d_model * max(1, len(self.cat_groups)) + len(self.num_idx)
        self.head = nn.Sequential(
            nn.Linear(head_in, d_ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ffn, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: (B, F)
        parts = []
        if self.cat_groups:
            tokens = torch.stack(
                [emb(x[:, g]) for emb, g in zip(self.cat_embed, self.cat_groups)],
                dim=1,
            )                                                  # (B, n_cat, D)
            for block in self.blocks:
                tokens = block(tokens)
            parts.append(self.norm(tokens).flatten(1))         # (B, n_cat*D)
        if self.num_idx:
            parts.append(self.num_norm(x[:, self.num_idx]))    # continuous bypass
        z = torch.cat(parts, dim=1)
        return torch.sigmoid(self.head(z))


class IntersampleAttention(nn.Module):
    """SAINT's second attention: rows attend to OTHER ROWS in the batch.

    The original formulation flattens each row's tokens to (F+1)*D and runs MHSA
    across the batch dimension; with F+1 = 29 and D = 64 that is a 1856-wide
    attention (~13.8 M parameters per block), which is disproportionate here and
    would inflate every federated round's payload. We keep the mechanism but add a
    bottleneck: each row is pooled to d_model, rows attend to rows in that space,
    and the result is broadcast back as a residual onto the row's tokens.
    This is a documented simplification of Somepalli et al. (2021).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        rows = self.norm(tokens.mean(dim=1)).unsqueeze(0)      # (1, B, D): batch as sequence
        ctx = self.attn(rows).squeeze(0)                       # (B, D)
        return tokens + ctx.unsqueeze(1)                       # broadcast residual


class SAINTBlock(nn.Module):
    """SAINT block = self-attention over FEATURES, then attention over ROWS."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.feature_block = TransformerBlock(d_model, n_heads, d_ffn, dropout)
        self.intersample = IntersampleAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_block(x)
        return self.dropout(self.intersample(x))


class SAINT(nn.Module):
    """SAINT (Somepalli et al. 2021), without contrastive pretraining.

    Same feature tokenizer as FT-Transformer, but each block adds intersample
    attention. CAVEAT: predictions therefore depend on the other rows in the
    batch — see the note in IntersampleAttention and the thesis Limitations.
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ffn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [SAINTBlock(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return torch.sigmoid(self.head(self.norm(tokens[:, 0])))


# ============================================================= LoRA injection

def add_lora(model: nn.Module, rank: int, alpha: float | None = None,
             include_head: bool = False) -> nn.Module:
    """Inject LoRA adapters STRUCTURALLY (by walking the module tree).

    Targets, for every transformer variant (FT, Simple, TabTransformer, SAINT):
      * Q, K, V projections inside every attention module (self- and intersample);
        the 'out' projection stays shared.
      * the FFN linear layers inside every transformer block.
    For an MLP: the hidden layers.
    Tokenizers, embeddings and the output head stay shared, so the FEDERATED
    backbone is identical across variants (verified in the tests).

    Walking the tree instead of isinstance-dispatch means new architectures are
    supported automatically, as long as they reuse MultiHeadSelfAttention /
    TransformerBlock.
    """
    if isinstance(model, MLP):
        linear_names = [n for n, m in model.net.named_children() if isinstance(m, nn.Linear)]
        if not include_head and linear_names:
            linear_names = linear_names[:-1]          # keep the output head shared
        for name in linear_names:
            model.net._modules[name] = LoRALinear(model.net._modules[name], rank, alpha)
        return model

    n_wrapped = 0
    for module in model.modules():
        # 1. attention: Q, K, V (out projection stays shared)
        if isinstance(module, MultiHeadSelfAttention):
            for proj in ("q", "k", "v"):
                layer = getattr(module, proj)
                if not isinstance(layer, LoRALinear):
                    setattr(module, proj, LoRALinear(layer, rank, alpha))
                    n_wrapped += 1
        # 2. FFN linears inside each transformer block
        if isinstance(module, TransformerBlock):
            for i, layer in enumerate(module.ffn):
                if isinstance(layer, nn.Linear) and not isinstance(layer, LoRALinear):
                    module.ffn[i] = LoRALinear(layer, rank, alpha)
                    n_wrapped += 1

    if include_head and isinstance(getattr(model, "head", None), nn.Linear):
        model.head = LoRALinear(model.head, rank, alpha)
        n_wrapped += 1

    if n_wrapped == 0:
        raise TypeError(f"add_lora: no adaptable layers found in {type(model).__name__}")
    return model


# ============================================================= backbone factory

# name -> class. Every transformer here shares the same block stack, so LoRA,
# backbone_state/adapter_state and FedAvg work identically for all of them.
TRANSFORMERS = {
    "transformer": FTTransformer,      # kept as the legacy alias for FT-Transformer
    "fttransformer": FTTransformer,
    "simple-transformer": SimpleTransformer,
    "tabtransformer": TabTransformer,
    "saint": SAINT,
}

BACKBONES = ["mlp", "mlp-wide", *TRANSFORMERS.keys()]


def _t_params(model_cfg: dict, name: str) -> dict:
    """Hyperparameters for a transformer: per-backbone block, else the shared one."""
    cfg = dict(model_cfg.get("transformer") or {})
    cfg.update(model_cfg.get(name) or {})
    return {"d_model": cfg.get("d_model", 64), "n_heads": cfg.get("n_heads", 4),
            "n_layers": cfg.get("n_layers", 2), "d_ffn": cfg.get("d_ffn", 128),
            "dropout": cfg.get("dropout", 0.1)}


def make_backbone(name: str, input_dim: int, model_cfg: dict | None = None) -> nn.Module:
    model_cfg = model_cfg or {}
    if name in ("mlp", "mlp-wide"):
        default_hidden = {"mlp": [64, 32], "mlp-wide": [128, 128]}[name]
        hidden = (model_cfg.get(name) or {}).get("hidden_dims", default_hidden)
        return MLP(input_dim, list(hidden))
    if name in TRANSFORMERS:
        p = _t_params(model_cfg, name)
        if name == "tabtransformer":
            # needs the one-hot column groups; supplied by the caller (feature spec)
            return TabTransformer(input_dim,
                                  model_cfg.get("cat_groups", []),
                                  model_cfg.get("num_idx", list(range(input_dim))), **p)
        return TRANSFORMERS[name](input_dim, **p)
    raise ValueError(f"make_backbone: unknown backbone '{name}' (have: {BACKBONES})")


def backbone_config(name: str, input_dim: int, model_cfg: dict | None = None) -> dict:
    model_cfg = model_cfg or {}
    if name in ("mlp", "mlp-wide"):
        default_hidden = {"mlp": [64, 32], "mlp-wide": [128, 128]}[name]
        hidden = (model_cfg.get(name) or {}).get("hidden_dims", default_hidden)
        return {"type": "mlp", "backbone": name, "input_dim": input_dim,
                "hidden_dims": list(hidden)}
    p = _t_params(model_cfg, name)
    mc = {"type": "transformer", "backbone": name, "n_features": input_dim, **p}
    if name == "tabtransformer":
        mc["cat_groups"] = model_cfg.get("cat_groups", [])
        mc["num_idx"] = model_cfg.get("num_idx", list(range(input_dim)))
    return mc


def build_from_config(mc: dict) -> nn.Module:
    """Rebuild a model from its saved blueprint (what the runtime container does)."""
    if mc.get("type") != "transformer":
        return MLP(mc["input_dim"], mc.get("hidden_dims", [64, 32]))
    name = mc.get("backbone", "transformer")
    cls = TRANSFORMERS.get(name)
    if cls is None:
        raise ValueError(f"build_from_config: unknown backbone '{name}'")
    p = {"d_model": mc.get("d_model", 64), "n_heads": mc.get("n_heads", 4),
         "n_layers": mc.get("n_layers", 2), "d_ffn": mc.get("d_ffn", 128),
         "dropout": mc.get("dropout", 0.1)}
    if name == "tabtransformer":
        return TabTransformer(mc["n_features"], mc.get("cat_groups", []),
                              mc.get("num_idx", []), **p)
    return cls(mc["n_features"], **p)


# ============================================================= backbone: shared, federated

def _norm_key(k: str) -> str:
    """Strip the '.linear.' level that LoRALinear inserts when wrapping an nn.Linear."""
    return k.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")


def backbone_state(model: nn.Module) -> dict:
    return {
        _norm_key(k): v.clone()
        for k, v in model.state_dict().items()
        if "lora_" not in k
    }


def load_backbone_state(model: nn.Module, state: dict) -> None:
    sd = model.state_dict()
    norm_to_actual = {_norm_key(k): k for k in sd if "lora_" not in k}
    for nk, v in state.items():
        if nk in norm_to_actual:
            sd[norm_to_actual[nk]] = v
    model.load_state_dict(sd)


def fed_avg(states: list[dict], weights: list[float] | None = None) -> dict:
    n = len(states)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = float(sum(weights))
        weights = [w / total for w in weights]
    return {
        key: sum(weights[i] * states[i][key] for i in range(n))
        for key in states[0]
    }


# ============================================================= adapters: private, per-site

def adapter_state(model: nn.Module) -> dict:
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def load_adapter_state(model: nn.Module, state: dict) -> None:
    sd = model.state_dict()
    sd.update(state)
    model.load_state_dict(sd)