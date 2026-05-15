"""Compat shim — registered via custom_imports.

PyTorch 2.6+ flipped `torch.load(..., weights_only=...)` to default True.
mmengine 0.10.7 checkpoints embed several non-tensor objects (HistoryBuffer,
numpy reconstructors, etc.) that aren't in PyTorch's default allowlist, so
`torch.load(...)` fails the moment we resume.

We trust our own checkpoint at runs/sam3_lora_v1/iter_*.pth, so we monkey-
patch `torch.load` to default `weights_only=False`. Callers that explicitly
pass `weights_only=True` still get the safe path.
"""

import datetime
import functools

import torch
import torch.distributed as dist

# -- 1. Make torch.load default to weights_only=False -------------------------
_original_load = torch.load


@functools.wraps(_original_load)
def _load_with_full_pickle(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)


torch.load = _load_with_full_pickle


# -- 2. Bump DDP collective timeout 10 min → 2 hours -------------------------
# Val takes ~70 min and the two ranks finish at slightly different times.
# Default NCCL collective timeout is 10 min in PyTorch 2.x; the faster rank
# enqueues the next allgather and SIGABRTs while waiting for the slower
# rank's val to finish. Force a long timeout at init.
_original_init_pg = dist.init_process_group


@functools.wraps(_original_init_pg)
def _init_pg_with_long_timeout(*args, **kwargs):
    if kwargs.get('timeout') is None:
        kwargs['timeout'] = datetime.timedelta(hours=2)
    return _original_init_pg(*args, **kwargs)


dist.init_process_group = _init_pg_with_long_timeout
