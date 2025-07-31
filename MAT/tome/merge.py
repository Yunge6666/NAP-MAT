# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def adjacent_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
        
    t = metric.shape[1]
    available_pairs = t - protected - 1
    r = min(r, available_pairs)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # calculate adjacent similarity
        metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        prev_tokens = metric_norm[:, protected:-1, :]    # [B, t-protected-1, C]
        next_tokens = metric_norm[:, protected+1:, :]    # [B, t-protected-1, C]
        adj_scores = (prev_tokens * next_tokens).sum(dim=-1)  # [B, t-protected-1]

        # prev_tokens = metric[:, protected:-1, :]
        # next_tokens = metric[:, protected+1:, :] 
        # prev_probs = F.softmax(prev_tokens, dim=-1)
        # next_probs = F.softmax(next_tokens, dim=-1)
        # kl_div_1 = (prev_probs * (prev_probs.log() - next_probs.log())).sum(dim=-1) 
        # kl_div_2 = (next_probs * (next_probs.log() - prev_probs.log())).sum(dim=-1)
        # adj_scores = -(kl_div_1 + kl_div_2) / 2  # [B, t-protected-1]

        _, dst_indices_rel = torch.topk(adj_scores, r, dim=-1)  # [B, r]
        dst_indices_rel = dst_indices_rel.sort(dim=-1)[0]  # sort
        
        dst_tokens = dst_indices_rel + protected  # [B, r] dst token absolute index
        src_tokens = dst_tokens + 1  # [B, r] src token absolute index
        
        B = adj_scores.shape[0]
        device = adj_scores.device
        
        if r > 1:
            # detect continuity: whether the difference between adjacent dst tokens is 1
            diffs = dst_tokens[:, 1:] - dst_tokens[:, :-1]  # [B, r-1]
            is_continuous = (diffs == 1)  # [B, r-1] 
            
            # mark the start of the interval: the first one is always the start, and the discontinuous places are also the start
            is_start = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=device),  # the first one is always the start
                ~is_continuous  # the discontinuous places are also the start
            ], dim=1)  # [B, r]
        else:
            is_start = torch.ones(B, r, dtype=torch.bool, device=device)
        
        start_indices = torch.arange(r, device=device).unsqueeze(0).expand(B, -1)  # [B, r]
        start_indices = torch.where(is_start, start_indices, 0)  # only keep the original index at the start position
        start_indices = start_indices.cummax(dim=1)[0]  # forward fill the maximum index
        
        final_dst_tokens = dst_tokens.gather(1, start_indices)  # [B, r]
        
        merge_info = {
            'src_tokens': src_tokens,           # [B, r] the index of the tokens to be merged
            'final_dst_tokens': final_dst_tokens,  # [B, r] the index of the corresponding merge target
            'protected': protected,
            't': t,
            'r': r
        }
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        B, T, C = x.shape
        r = merge_info['r']
        
        if r == 0:
            return x
        
        src_tokens = merge_info['src_tokens']           # [B, r]
        final_dst_tokens = merge_info['final_dst_tokens']  # [B, r]
        
        batch_indices = torch.arange(B, device=x.device)[:, None]  # [B, 1]
        
        src_values = x[batch_indices, src_tokens]
        
        # merge the src values to the corresponding dst positions
        x.scatter_reduce_(1, final_dst_tokens.unsqueeze(-1).expand(-1, -1, C), src_values, reduce=mode)
        
        # create keep_mask, remove src_tokens
        keep_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        keep_mask[batch_indices, src_tokens] = False
        
        return x[keep_mask].view(B, T - r, C)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        B, merged_T, C = x.shape
        t = merge_info['t']
        r = merge_info['r']
        
        if r == 0:
            return x
            
        out = torch.zeros(B, t, C, device=x.device, dtype=x.dtype)
        
        src_tokens = merge_info['src_tokens']           # [B, r]
        final_dst_tokens = merge_info['final_dst_tokens']  # [B, r]
        
        batch_indices = torch.arange(B, device=x.device)[:, None]  # [B, 1]
        
        # create keep_mask to identify which tokens are kept
        keep_mask = torch.ones(B, t, dtype=torch.bool, device=x.device)
        keep_mask[batch_indices, src_tokens] = False
        
        # restore the kept tokens to the original position
        keep_indices = torch.arange(t, device=x.device)[None, :].expand(B, -1)[keep_mask].view(B, -1)
        out.scatter_(1, keep_indices.unsqueeze(-1).expand(-1, -1, C), x)
        
        target_values = out[batch_indices, final_dst_tokens]  # [B, r, C]
        out[batch_indices, src_tokens] = target_values
        
        return out

    return merge, unmerge
