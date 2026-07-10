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


# ============================================================= LoRA injection

def add_lora(model: nn.Module, rank: int, alpha: float | None = None,
             include_head: bool = False) -> nn.Module:
    """Replace linear layers with LoRALinear in place.

    MLP: Adapts the hidden layers only.
    FTTransformer: Applies LoRA to Q, K, V projections and the Feed-Forward Network (FFN) 
    linear layers inside each block, exactly matching Klein et al. (2026)[cite: 167]. 
    The 'out' projection, tokenizer, and head stay shared.
    """
    if isinstance(model, MLP):
        linear_names = [n for n, m in model.net.named_children() if isinstance(m, nn.Linear)]
        if not include_head and linear_names:
            linear_names = linear_names[:-1]     # Output-Head nicht adaptieren
        for name in linear_names:
            model.net._modules[name] = LoRALinear(model.net._modules[name], rank, alpha)
            
    elif isinstance(model, FTTransformer):
        for block in model.blocks:
            # 1. Q, K, V Projektionen in der Attention adaptieren (out bleibt unberührt) [cite: 167]
            attn = block.attn
            for proj in ("q", "k", "v"):
                setattr(attn, proj, LoRALinear(getattr(attn, proj), rank, alpha))
            
            # 2. Lineare Schichten im Feed-Forward-Netzwerk (FFN) adaptieren [cite: 167]
            for i, layer in enumerate(block.ffn):
                if isinstance(layer, nn.Linear):
                    block.ffn[i] = LoRALinear(layer, rank, alpha)
    else:
        raise TypeError(f"add_lora: unsupported model type {type(model)}")
    return model


# ============================================================= backbone factory

def make_backbone(name: str, input_dim: int, model_cfg: dict | None = None) -> nn.Module:
    model_cfg = model_cfg or {}
    if name in ("mlp", "mlp-wide"):
        default_hidden = {"mlp": [64, 32], "mlp-wide": [128, 128]}[name]
        hidden = (model_cfg.get(name) or {}).get("hidden_dims", default_hidden)
        return MLP(input_dim, list(hidden))
    if name == "transformer":
        t = model_cfg.get("transformer") or {}
        return FTTransformer(input_dim, t.get("d_model", 64), t.get("n_heads", 4),
                             t.get("n_layers", 2), t.get("d_ffn", 128),
                             dropout=t.get("dropout", 0.1))
    raise ValueError(f"make_backbone: unknown backbone '{name}'")


def backbone_config(name: str, input_dim: int, model_cfg: dict | None = None) -> dict:
    model_cfg = model_cfg or {}
    if name in ("mlp", "mlp-wide"):
        default_hidden = {"mlp": [64, 32], "mlp-wide": [128, 128]}[name]
        hidden = (model_cfg.get(name) or {}).get("hidden_dims", default_hidden)
        return {"type": "mlp", "backbone": name, "input_dim": input_dim, "hidden_dims": list(hidden)}
    t = model_cfg.get("transformer") or {}
    return {"type": "transformer", "backbone": name, "n_features": input_dim,
            "d_model": t.get("d_model", 64), "n_heads": t.get("n_heads", 4),
            "n_layers": t.get("n_layers", 2), "d_ffn": t.get("d_ffn", 128),
            "dropout": t.get("dropout", 0.1)}


def build_from_config(mc: dict) -> nn.Module:
    if mc.get("type") == "transformer":
        return FTTransformer(mc["n_features"], mc.get("d_model", 64), mc.get("n_heads", 4),
                             mc.get("n_layers", 2), mc.get("d_ffn", 128),
                             dropout=mc.get("dropout", 0.1))
    return MLP(mc["input_dim"], mc.get("hidden_dims", [64, 32]))


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