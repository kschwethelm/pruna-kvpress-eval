"""Shared utilities for benchmark scripts."""

import torch

_printed = False


def select_attn_implementation() -> str:
    """Return the best available flash attention implementation for the current GPU.

    FA3 requires Hopper (compute capability >= 9.0, e.g. H100/H200).
    FA2 is used on Ampere and older (A100, etc.).
    Prints the selection once per process.
    """
    global _printed
    if not torch.cuda.is_available():
        impl = "sdpa"
        device_name = "CPU"
    else:
        major, _ = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name(0)
        impl = "flash_attention_3" if major >= 9 else "flash_attention_2"
    if not _printed:
        print(f"[attn] GPU={device_name}, selected attn_implementation={impl}")
        _printed = True
    return impl
