"""Forward Transformer Engine's FA3 import to the top-level FlashAttention module."""

try:
    from flash_attn_interface import *  # noqa: F401,F403
except ImportError as exc:
    raise ImportError(
        "Failed to import top-level flash_attn_interface required by the local "
        "flash_attn_3 compatibility shim."
    ) from exc
