"""Memory safety for 30GB Kaggle machines.

Every large loop in the pipeline processes data in chunks. Before each chunk we look at the
machine's *available* memory (not this process's RSS, which includes allocator-retained pages
the kernel can reclaim). If headroom is low we first collect garbage, then halve the chunk size:
the run gets slower instead of being OOM-killed.
"""
import gc
import os

LOW_GB = float(os.environ.get('BER_MEM_LOW_GB', 6.0))   # headroom threshold, overridable


def available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float('inf')


def next_chunk(chunk, floor, log=None, tag=''):
    if available_gb() >= LOW_GB:
        return chunk
    gc.collect()
    avail = available_gb()
    if avail < LOW_GB and chunk > floor:
        chunk = max(floor, chunk // 2)
        if log is not None:
            log(f'  memguard[{tag}]: only {avail:.1f}GB free -> chunk size reduced to {chunk:,}')
    return chunk
