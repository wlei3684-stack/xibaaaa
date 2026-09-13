"""Paired cross scan using the upstream optimized cross-scan/merge kernels."""
import torch
try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except ImportError:
    from csm_triton import cross_scan_fn, cross_merge_fn


def pair_scan(template, search, force_torch=False):
    """Return B,4,C,Lt+Ls, in upstream row/column/reverse-row/reverse-column order."""
    ts = cross_scan_fn(template, force_torch=force_torch)
    ss = cross_scan_fn(search, force_torch=force_torch)
    # Forward directions T->S; reverse directions S->T. No per-token loop.
    forward = torch.cat((ts[:, :2], ss[:, :2]), dim=-1)
    reverse = torch.cat((ss[:, 2:], ts[:, 2:]), dim=-1)
    return torch.cat((forward, reverse), dim=1)


def pair_merge(sequences, template_hw, search_hw, force_torch=False):
    """Split at direction-specific boundaries, then restore each spatial grid."""
    ht, wt = template_hw
    hs, ws = search_hw
    lt, ls = ht * wt, hs * ws
    b, k, c, length = sequences.shape
    if k != 4 or length != lt + ls:
        raise ValueError('Expected four scan directions and length Lt + Ls')
    ts = torch.cat((sequences[:, :2, :, :lt], sequences[:, 2:, :, ls:]), dim=1)
    ss = torch.cat((sequences[:, :2, :, lt:], sequences[:, 2:, :, :ls]), dim=1)
    template = cross_merge_fn(ts.view(b, 4, c, ht, wt), force_torch=force_torch)
    search = cross_merge_fn(ss.view(b, 4, c, hs, ws), force_torch=force_torch)
    return template.view(b, c, ht, wt), search.view(b, c, hs, ws)
