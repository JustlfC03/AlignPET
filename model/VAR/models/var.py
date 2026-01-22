import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import model.VAR.dist as dist
from model.VAR.models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from model.VAR.models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from model.VAR.models.vqvae import VQVAE, VectorQuantizer2
import torchvision.models as emodels


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class VAR(nn.Module):
    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l,
                                                                                                  -1) + lvl_pos[:,
                                                                                                        :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


import torch.nn.functional as F


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None:
        attn.add_(attn_mask)
    if dropout_p > 0:
        attn = F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
    else:
        attn = attn.softmax(dim=-1)
    return attn @ value



# class CrossAttentionAR(nn.Module):
#     def __init__(
#             self, block_idx, dim=1024, num_heads=16, proj_drop=0., attn_l2_norm=False,
#     ):
#         super(CrossAttentionAR, self).__init__()

#         self.attn_l2_norm = attn_l2_norm
#         self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, dim // num_heads
#         self.scale = 0.25 / math.sqrt(self.head_dim)

#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()

#         self.mat_q = nn.Linear(dim, dim, bias=False)
#         self.mat_kv = nn.Linear(dim, dim * 2, bias=False)
#         self.q_bias, self.v_bias = nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))
#         self.register_buffer('zero_k_bias', torch.zeros(dim))

#     def forward(self, f_d, f_var, attn_bias=None):
#         B, L, C = f_d.shape
#         f_q = F.linear(input=f_d, weight=self.mat_q.weight, bias=self.q_bias).view(B, L, 1, self.num_heads,
#                                                                                    self.head_dim)
#         f_kv = F.linear(input=f_var, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(B,
#                                                                                                                       L,
#                                                                                                                       2,
#                                                                                                                       self.num_heads,
#                                                                                                                       self.head_dim)


#         q = f_q.permute(2, 0, 3, 1, 4)
#         k, v = f_kv.permute(2, 0, 3, 1, 4).unbind(dim=0);


#         dropout_p = 0.0


#         oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias,
#                             dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

#         return self.proj_drop(self.proj(oup))

#     def extra_repr(self) -> str:
#         return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'
class CrossAttentionAR(nn.Module):
    def __init__(
        self, block_idx, dim=1024, num_heads=16, proj_drop=0., attn_l2_norm=False
    ):
        super().__init__()
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # 标准缩放因子

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # 可学习的偏置
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.k_bias = nn.Parameter(torch.zeros(dim))  # 替代 zero_k_bias
        self.v_bias = nn.Parameter(torch.zeros(dim))

        # 归一化层
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        # 投影层
        self.mat_q = nn.Linear(dim, dim, bias=False)
        self.mat_k = nn.Linear(dim, dim, bias=False)  # 显式定义 Key 投影
        self.mat_v = nn.Linear(dim, dim, bias=False)  # 显式定义 Value 投影

    def forward(self, f_d, f_var, attn_bias=None):
        B, L, C = f_d.shape

        # 归一化 + 投影
        q = self.norm_q(f_d)
        k = self.norm_k(f_var)
        v = self.norm_v(f_var)

        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.mat_k(k).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.mat_v(v).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        dropout_p = 0.0
        oup = slow_attn(q, k, v, self.scale, attn_mask=attn_bias, dropout_p=dropout_p)

        # 合并头 + 投影
        oup = oup.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(oup))

class VARrecon(nn.Module):
    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.MRIencoder = emodels.resnet50()
        num_ftrs = self.MRIencoder.fc.in_features
        self.MRIencoder.fc = nn.Linear(num_ftrs, 1024)
        the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MRIencoder = self.MRIencoder.to(the_device)

        self.weight = nn.Parameter(torch.tensor(0.5))
        self.parassos = nn.Parameter(torch.rand(1, 1, 1024))

        # self.cross_attn_blocks = nn.ModuleList([
        #     CrossAttentionAR(block_idx, embed_dim, attn_l2_norm=attn_l2_norm)
        #     for block_idx in range(int(depth // 8))
        # ])
        self.word_embed2 = nn.Linear(self.Cvae, self.C)


    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
            self, inpMRI,
            g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
            more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        B = inpMRI.shape[0]

        f_hats = []
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed);
            rng = self.rng

        uncond = torch.full_like(inpMRI, fill_value=0)
        uncondf = torch.cat((inpMRI, uncond), dim=0)
        fmri = self.MRIencoder(uncondf)

        sos = fmri
        cond_BD = sos

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l,
                                                                                                  -1) + lvl_pos[:,
                                                                                                        :self.first_l]
        #next_token_map2 = sos.unsqueeze(1)

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for index in range(len(self.blocks)):
                b = self.blocks[index]
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                # if index <=8:
                #     x = 0.5*x +0.5*next_token_map2
                #     x = next_token_map2
                #     x = self.cross_attn_blocks[index](next_token_map2, x)
                
            logits_BlV = self.get_logits(x, cond_BD)
            

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            #idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            # print(idx_Bl.shape)

            #h_BChw = idx_Bl

            #h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
           
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
            h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            #print(h_BChw.shape,logits_BlV.shape)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.vae_quant_proxy[0].get_next_autoregressive_input(si,
                                                                                                len(self.patch_nums),
                                                                                                f_hat, h_BChw)
            f_hats.append(h_up.unsqueeze(1))
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

                # next_token_map2 = x_BLCv_wo_first_l[:, cur_L - 1:cur_L - 1 + self.patch_nums[si + 1] ** 2]
                # next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:,
                #                                                      cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                # next_token_map2 = next_token_map2.repeat(2, 1, 1)

        for b in self.blocks: b.attn.kv_caching(False)

        f_out = torch.cat((f_hats), dim=1)
        #print(f_hat.shape)
        #return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5), f_out
        return self.vae_proxy[0].fhat_to_img(f_hat), f_out

    def forward(self, x_BLCv_wo_first_l: torch.Tensor, inpMRI,x_BLCv_wo_first_l2
                ):
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        the_condition = self.word_embed(x_BLCv_wo_first_l.float())
        #the_mri_condition = self.word_embed2(x_BLCv_wo_first_l2.float())

        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            fmri = self.MRIencoder(inpMRI)
            sos = fmri
            cond_BD = sos
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, the_condition), dim=1)
                the_condition = torch.cat((sos, the_condition), dim=1)
                #print(the_condition.shape)
                #the_mri_condition = torch.cat((sos, the_mri_condition), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        #print(attn_bias.shape)
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        #print(cond_BD_or_gss.shape)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            #print(x_BLC.shape)
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            # if i <=8:
            #     x_BLC = 0.5*x_BLC+0.5*the_condition
            #     x_BLC = self.cross_attn_blocks[i](the_condition,x_BLC,attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        print(x_BLC.shape)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return None,x_BLC,None  # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (
                    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (
                    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'

