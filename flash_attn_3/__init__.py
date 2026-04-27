"""Compatibility shim for environments where FlashAttention-3 installs
`flash_attn_interface` as a top-level module but Transformer Engine imports
`flash_attn_3.flash_attn_interface`.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
