"""Minimal transformer model configurations for KataGo nano training."""

from typing import Dict, Any

ModelConfig = Dict[str, Any]


def get_version(config: ModelConfig):
    return config["version"]


def get_num_bin_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 22
    elif version == 101 or version == 102:
        return 22
    else:
        assert False


def get_num_global_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15:
        return 19
    elif version == 101 or version == 102:
        return 39
    else:
        assert False


def make_config(num_layers, hidden_size, num_heads, ffn_dim=None, num_scorebeliefs=8, version=15):
    """Create a model config from minimal parameters.

    Args:
        num_layers: Number of transformer blocks.
        hidden_size: Hidden dimension (trunk channels).
        num_heads: Number of attention heads.
        ffn_dim: SwiGLU FFN intermediate dimension. Default: hidden_size * 8 // 3.
        num_scorebeliefs: Number of score belief mixtures. Default: 8.
        version: Data format version. Default: 15.
    """
    if ffn_dim is None:
        ffn_dim = hidden_size * 8 // 3
    return {
        "version": version,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "num_scorebeliefs": num_scorebeliefs,
    }


def migrate_config(old: ModelConfig) -> ModelConfig:
    """Convert old-format config (with trunk_num_channels etc.) to new minimal format."""
    if "hidden_size" in old:
        old = dict(old)
        # Remove legacy fields that are no longer supported
        for legacy_key in ("use_ape", "ape", "pos_enc", "rpe",
                           "stem_d4", "stem_norm", "stem_init_aligned",
                           "stem"):
            old.pop(legacy_key, None)
        return old
    return make_config(
        num_layers=len(old["block_kind"]),
        hidden_size=old["trunk_num_channels"],
        num_heads=old["transformer_heads"],
        ffn_dim=old.get("transformer_ffn_channels", old["trunk_num_channels"] * 8 // 3),
        num_scorebeliefs=old.get("num_scorebeliefs", 8),
        version=old.get("version", 15),
    )


# ---------------------------------------------------------------------------
# Predefined model configs
# ---------------------------------------------------------------------------

# ~5M params — ViT-Ti
b12c192 = make_config(12, 192, 3, ffn_dim=512)

# ~22M params — ViT-S
b12c384 = make_config(12, 384, 6, ffn_dim=1024)

# ~90M params — ViT-B
b12c768 = make_config(12, 768, 12, ffn_dim=2048)
b24c512 = make_config(24, 512, 8, ffn_dim=1536)

# ~330M params — ViT-L
b24c1024 = make_config(24, 1024, 16, ffn_dim=3072)

config_of_name = {
    "b12c192": b12c192,
    "b12c384": b12c384,
    "b24c512": b24c512,
    "b12c768": b12c768,
    "b24c1024": b24c1024,
}
