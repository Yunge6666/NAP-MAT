# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# mae: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer


from .deit import DiffRateBlock, DiffRateAttention

from DiffRate.utils import ste_min



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

def make_diffrate_class(transformer_class):
    class DiffRateVisionTransformer(transformer_class):
        def forward(self, x, return_flop=True) -> torch.Tensor:
            B = x.shape[0]
            self._diffrate_info["size"] = torch.ones([B,self.patch_embed.num_patches+1,1], device=x.device)
            self._diffrate_info["mask"] =  torch.ones((B,self.patch_embed.num_patches+1),device=x.device)
            self._diffrate_info["prune_kept_num"] = []
            self._diffrate_info["merge_kept_num"] = []
            x = super().forward(x)
            if return_flop:
                if self.training:
                    flops = self.calculate_flop_training()
                else:
                    flops = self.calculate_flop_inference()
                return x, flops
            else:
                return x
            
        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            x = self.patch_embed(x)

            T = x.shape[1]

            cls_tokens = self.cls_token.expand(B, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            cls_tokens = x[:, :1, :] 
            patches = x[:, 1:, :]     
            
            if hasattr(self.patch_embed, 'img_size') and hasattr(self.patch_embed, 'patch_size'):
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                if isinstance(img_size, int):
                    img_size = (img_size, img_size)
                if isinstance(patch_size, int):
                    patch_size = (patch_size, patch_size)
                
                H = img_size[0] // patch_size[0]
                W = img_size[1] // patch_size[1]
                
                patches_2d = patches.view(B, H, W, -1)  # [B, H, W, C]
                patches_reordered = tensor_to_gilbert_path(patches_2d, cache=_global_gilbert_cache)  # [B, H*W, C]
                
                x = torch.cat([cls_tokens, patches_reordered], dim=1)

            for blk in self.blocks:
                x = blk(x)

            if self.global_pool:
                if self.training:
                    mask = self._diffrate_info["mask"][...,None]  # [B, N, 1]
                    num = (self._diffrate_info["size"] * mask)[:, 1:, :].sum(dim=1) # [B,1]
                    x = (x * self._diffrate_info["size"] * mask)[:, 1:, :].sum(dim=1) / num.detach()
                    outcome = self.fc_norm(x)
                else:
                    T = self._diffrate_info["size"][:, 1:, :].sum(dim=1)
                    if self._diffrate_info["size"] is not None:
                        x = (x * (self._diffrate_info["size"]))[:, 1:, :].sum(dim=1) / T
                    else:
                        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                    outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                x = self.pre_logits(x)
                outcome = x[:, 0]            

            return outcome
        
        def parameters(self, recurse=True):
            # original network parameter
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    continue
                params.append(m)
            return iter(params)    
        
        def arch_parameters(self):
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    params.append(m)
            return iter(params)    
    

        def get_kept_num(self):
            prune_kept_num = []
            merge_kept_num = []
            for block in self.blocks:
                prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_kept_num.append(int(block.merge_ddp.kept_token_number))
            return prune_kept_num, merge_kept_num
        
        def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
            assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
            for block, prune_kept_number, merge_kept_number in zip(self.blocks, prune_kept_numbers, merge_kept_numbers):
                block.prune_ddp.kept_token_number = prune_kept_number
                block.merge_ddp.kept_token_number = merge_kept_number
        
        def calculate_flop_training(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for prune_kept_number, merge_kept_number in zip(self._diffrate_info["prune_kept_num"],self._diffrate_info["merge_kept_num"]):
                    # translate fp16 to fp32 for stable training
                    prune_kept_number = prune_kept_number.float()     
                    merge_kept_number = merge_kept_number.float()
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1, device=self.blocks[0].prune_ddp.selected_probability.device)
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    prune_kept_number = block.prune_ddp.kept_token_number
                    merge_kept_number = block.merge_ddp.kept_token_number
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = ste_min(N, prune_kept_number, merge_kept_number)
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops
        

    return DiffRateVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False,prune_granularity=1, merge_granularity=1
):
    """
    Applies DiffRate to this transformer.
    """
    DiffRateVisionTransformer = make_diffrate_class(model.__class__)

    model.__class__ = DiffRateVisionTransformer
    model._diffrate_info = {
        "size": None,
        "mask": None,           # only for training
        "source": None,
        "class_token": model.cls_token is not None,
        "trace_source": trace_source,
    }

    block_index = 0
    # non_compressed_block_index = [0]
    non_compressed_block_index = [0, len(model.blocks)-1]
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DiffRateBlock
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(model.patch_embed.num_patches, model.patch_embed.num_patches+1, model.patch_embed.num_patches+1)
            else:
                module.introduce_diffrate(model.patch_embed.num_patches, prune_granularity, merge_granularity)
            block_index += 1
            module._diffrate_info = model._diffrate_info
        elif isinstance(module, Attention):
            module.__class__ = DiffRateAttention