# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple
import time
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.merge import bipartite_soft_matching, adjacent_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r

def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int, result):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = sgn(ax), sgn(ay)
    dbx, dby = sgn(bx), sgn(by)

    if h == 1 or w == 1:
        if h == 1:
            for _ in range(w):
                result.append((x, y))
                x, y = x + dax, y + day
        elif w == 1:
            for _ in range(h):
                result.append((x, y))
                x, y = x + dbx, y + dby
        return

    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if w2 % 2 and w > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        generate2d(x, y, ax2, ay2, bx, by, result)
        generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, result)
    else:
        if h2 % 2 and h > 2:
            bx2, by2 = bx2 + dbx, by2 + dby
        generate2d(x, y, bx2, by2, ax2, ay2, result)
        generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, result)
        generate2d(x + (ax - dax) + (bx2 - dbx),
                   y + (ay - day) + (by2 - dby),
                   -bx2, -by2, -(ax - ax2), -(ay - ay2), result)

def gilbert2d(width, height):
    result = []
    if width >= height:
        generate2d(0, 0, width, 0, 0, height, result)
    else:
        generate2d(0, 0, 0, height, width, 0, result)
    return result

class GilbertPathCache:
    def __init__(self):
        self.cache = {}
        
    def get_or_create_path(self, H, W):
        key = (H, W)
        if key not in self.cache:
            path = gilbert2d(W, H)
            
            forward_map = torch.zeros((H, W), dtype=torch.long)
            reverse_map = torch.zeros((H * W, 2), dtype=torch.long)
            
            for idx, (x, y) in enumerate(path[:H*W]):
                if y < H and x < W:
                    forward_map[y, x] = idx
                    reverse_map[idx, 0] = y
                    reverse_map[idx, 1] = x
            
            self.cache[key] = {
                'path': path,
                'forward_map': forward_map,
                'reverse_map': reverse_map,
                'H': H,
                'W': W
            }
        
        return self.cache[key]
    
    def precompute_paths(self, resolutions):
        for H, W in resolutions:
            self.get_or_create_path(H, W)
    
    def clear_cache(self):
        self.cache.clear()

_global_gilbert_cache = GilbertPathCache()

def tensor_to_gilbert_path(x, cache=None):
    """
    Args:
        x: Input tensor, shape (B, H, W, C)
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        Reordered tensor, shape (B, H*W, C)
    """
    B, H, W, C = x.shape
    device = x.device
    if cache is None:
        cache = _global_gilbert_cache
    
    path_info = cache.get_or_create_path(H, W)
    reverse_map = path_info['reverse_map'].to(device)  # (H*W, 2)
    
    y_indices = reverse_map[:, 0]  # (H*W,)
    x_indices = reverse_map[:, 1]  # (H*W,)
    
    gilbert_tensor = x[:, y_indices, x_indices, :]  # (B, H*W, C)
    
    return gilbert_tensor

def gilbert_tensor_to_2d(x, H, W, cache=None):
    """
    Args:
        x: Gilbert sequence tensor, shape (B, H*W, C)
        H: Target height
        W: Target width
        cache: Optional GilbertPathCache instance, use global cache if None
    Returns:
        2D layout tensor, shape (B, H, W, C)
    """
    B, N, C = x.shape
    device = x.device
    
    if cache is None:
        cache = _global_gilbert_cache
    
    path_info = cache.get_or_create_path(H, W)
    reverse_map = path_info['reverse_map'].to(device)  # (H*W, 2)
    
    output_2d = torch.zeros((B, H, W, C), dtype=x.dtype, device=device)
    
    valid_n = min(N, H * W)
    if valid_n > 0:
        y_indices = reverse_map[:valid_n, 0]  # (valid_n,)
        x_indices = reverse_map[:valid_n, 1]  # (valid_n,)
        
        output_2d[:, y_indices, x_indices, :] = x[:, :valid_n, :]
    
    return output_2d

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # obtain the current block index
            current_block_index = self._tome_info["current_block_index"]
            self._tome_info["current_block_index"] += 1
            
            # select the merge method according to the block index
            if current_block_index % 2 == 0:
                merge, _ = adjacent_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                # merge, _ = bipartite_soft_matching(
                #     metric,
                #     r,
                #     self._tome_info["class_token"],
                #     self._tome_info["distill_token"],
                # )
            else:
                merge, _ = adjacent_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                # merge, _ = bipartite_soft_matching(
                #     metric,
                #     r,
                #     self._tome_info["class_token"],
                #     self._tome_info["distill_token"],
                # )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - reorder the patch tokens according to the Gilbert curve at the beginning of the model
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["current_block_index"] = 0  # reset block index counter at the beginning of each forward

            return super().forward(*args, **kwdargs)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            cls_tokens = x[:, :1, :]  # [B, 1, C] class token
            patches = x[:, 1:, :]     # [B, H*W, C] patch tokens
            
            if hasattr(self.patch_embed, 'img_size') and hasattr(self.patch_embed, 'patch_size'):
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                if isinstance(img_size, int):
                    img_size = (img_size, img_size)
                if isinstance(patch_size, int):
                    patch_size = (patch_size, patch_size)
                
                H = img_size[0] // patch_size[0]
                W = img_size[1] // patch_size[1]
                
                if patches.shape[1] == H * W:  # 确保维度匹配
                    patches_2d = patches.view(B, H, W, -1)  # [B, H, W, C]
                    patches_reordered = tensor_to_gilbert_path(patches_2d, cache=_global_gilbert_cache)  # [B, H*W, C]
                    x = torch.cat([cls_tokens, patches_reordered], dim=1)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            return x[:, 0]

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r =13
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "current_block_index": 0,  # add block index counter
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
