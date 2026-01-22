import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

import pywt
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from natsort import decoder
from model.FuseUnet.Decoder import MyDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from generative.losses import PerceptualLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from typing import Union, Type, List, Tuple

from model.HLVAE.model import HLVAE
from evaluation.eval import save_tensor_to_nii, calculate_nmse_nmae
from dataloader.dataloader3D import myDataset3D

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op

import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from evaluation.eval import calculate_psnr_ssim
from model.our.GAN_loss import NLayerDiscriminator3D

import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 只沿 depth 轴做 attention（最安全）
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # 生成 qkv
        qkv = self.qkv(x)  # (B, 3C, D, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # each: (B, C, D, H, W)

        # 重塑：沿 depth 轴做 attention → 把每个 (H, W) 位置上的 D 个体素作为序列
        # (B, C, D, H, W) -> (B, H, W, num_heads, D, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, D, H, W).permute(0, 4, 5, 1, 3, 2)  # (B, H, W, H_h, D, C/H)
        k = k.view(B, self.num_heads, self.head_dim, D, H, W).permute(0, 4, 5, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, D, H, W).permute(0, 4, 5, 1, 3, 2)

        # 计算 attention: (B, H, W, H_h, D, C/H) @ (B, H, W, H_h, C/H, D) -> (B, H, W, H_h, D, D)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # 加权求和: (B, H, W, H_h, D, D) @ (B, H, W, H_h, D, C/H) -> (B, H, W, H_h, D, C/H)
        out = attn @ v

        # 恢复形状: (B, H, W, H_h, D, C/H) -> (B, C, D, H, W)
        out = out.permute(0, 3, 5, 4, 1, 2).reshape(B, C, D, H, W)

        # 投影 + 残差
        out = self.project_out(out)
        return out + x  # 残差连接

class CrossAttentionFusion(nn.Module):
    def __init__(self, stage_channels, guidance_channels=64, d_model=64, reduction_ratio=4):
        super().__init__()
        self.reduced_channels = max(8, stage_channels // reduction_ratio)

        # 通道压缩/扩展
        self.reduce_conv = nn.Conv3d(stage_channels, self.reduced_channels, kernel_size=1)
        self.expand_conv = nn.Conv3d(self.reduced_channels, stage_channels, kernel_size=1)

        # 注意力层
        self.query_linear = nn.Linear(self.reduced_channels, d_model)
        self.key_linear = nn.Linear(guidance_channels, d_model)
        self.value_linear = nn.Linear(guidance_channels, d_model)
        self.out_linear = nn.Linear(d_model, self.reduced_channels)

        self.d_model = d_model

    def forward(self, stage_output, guidance_vector):
        B, C, D, H, W = stage_output.shape

        guidance_pooled = guidance_vector.mean(dim=[2, 3, 4])
        guidance_pooled = guidance_pooled.unsqueeze(1)

        # 2. 减少阶段输出的通道数
        reduced_stage = self.reduce_conv(stage_output)

        reduced_stage_flat = reduced_stage.view(B, self.reduced_channels, -1).transpose(1, 2)

        query = self.query_linear(reduced_stage_flat)
        key = self.key_linear(guidance_pooled)
        value = self.value_linear(guidance_pooled)

        attn_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_model)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_output = torch.matmul(attn_weights, value)
        output = self.out_linear(attn_output)
        output = output.transpose(1, 2).view(B, self.reduced_channels, D, H, W)
        output = self.expand_conv(output)
        return stage_output + output


import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIGuidedAttentionGate3D(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # 如果你坚持要 [0,1] 门控

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)  # 输出注意力权重图
        return x


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))

            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        self.var_condition = nn.ModuleList([
            CrossAttentionFusion(32, d_model=64),  # Stage1: 32通道
            CrossAttentionFusion(64, d_model=64),  # Stage2: 64通道
            CrossAttentionFusion(128, d_model=128),  # Stage3: 128通道
            CrossAttentionFusion(256, d_model=128),  # Stage4: 256通道
            CrossAttentionFusion(320, d_model=256),  # Stage5: 320通道
            CrossAttentionFusion(320, d_model=256)  # Stage6: 320通道
        ])

        self.self_attention = nn.ModuleList([
                                        AxialAttention3D(128),
                                         ])

        self.attention = nn.ModuleList([MRIGuidedAttentionGate3D(1, out_channel=32),
                                        MRIGuidedAttentionGate3D(32, out_channel=64, stride=2),
                                        MRIGuidedAttentionGate3D(64, out_channel=128, stride=2),
                                        MRIGuidedAttentionGate3D(128, out_channel=256, stride=2),
                                        MRIGuidedAttentionGate3D(256, out_channel=320, stride=2),
                                        MRIGuidedAttentionGate3D(320, out_channel=320, stride=2),
                                      ])


        # self.attention2 = nn.ModuleList([MRIGuidedAttentionGate3D(1, out_channel=32),
        #                                 MRIGuidedAttentionGate3D(32, out_channel=64, stride=2),
        #                                 MRIGuidedAttentionGate3D(64, out_channel=128, stride=2),
        #                                 MRIGuidedAttentionGate3D(128, out_channel=256, stride=2),
        #                                 MRIGuidedAttentionGate3D(256, out_channel=320, stride=2),
        #                                 MRIGuidedAttentionGate3D(320, out_channel=320, stride=2),
        #                               ])
        

        self.last_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(320, 320, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(320),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(320, 320, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(320),
                nn.ReLU(inplace=True)
            ),
        ])



    def forward(self, x, var_condition, MRIwenli,ceb):
        ret = []
        weight = MRIwenli
        for i in range(len(self.stages)):
            
            s = self.stages[i]
            v = self.var_condition[i]
            x = s(x)
            x = v(x, var_condition)
            
            at = self.attention[i]
            
            weight = at(weight)
            x = weight * x + x


            # at2 = self.attention2[i]
            # ceb = at2(ceb)
            # x = ceb * x + x

            if i == 2:
                sa = self.self_attention[0]
                x = sa(x)

            last_conv = self.last_conv[i]
            x = last_conv(x)

            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output



mod = 1


class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        if mod:
            self.decoder = MyDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                     nonlin_first=nonlin_first)
        else:
            self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       nonlin_first=nonlin_first)
        self.last_conv = nn.Sequential(
                nn.BatchNorm3d(1),
                nn.ReLU(inplace=True),
                nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False) )

    def forward(self, x, var_condition, wenli,ceb):
        skips = self.encoder(x, var_condition, wenli,ceb)
        if mod:
            return self.last_conv(self.decoder(x, skips))
        else:
            return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


def train_trans(train_dataloader, val_dataloader, epochs, device, resume, HL_VAE_path=None, not_xiaobo=False,
                only_eval=False):
    model = HLVAE().to(device)
    if HL_VAE_path is None:
        print('The save path for HL_VAE has not been set.')
    translate_model_save_path = HL_VAE_path

    if resume:
        model.load_state_dict(torch.load(translate_model_save_path))

    optimizer_trans = torch.optim.AdamW(model.parameters(), lr=0.0001)

    max_psnr = 0
    max_ssim = 0
    min_nmse = float('inf')
    min_nmae = float('inf')

    for epoch in range(epochs):
        if not only_eval:
            loss_save = 0
            model.train()
            train_loader_with_progress = tqdm(train_dataloader,
                                              desc=f'Epoch [{epoch + 1}/{epochs}]',
                                              leave=False)  # leave=False 表示结束后清除进度条
            for (inpmri, gtpet, MCimages, xiaobo_pet, wenli) in train_loader_with_progress:
                inpmri, gtpet, MCimages, xiaobo_pet, wenli = inpmri.to(device), gtpet.to(
                    device), MCimages.to(device), xiaobo_pet.to(device), wenli.to(device)

                # GT = xiaobo_pet
                GT = wenli
                if not_xiaobo:
                    GT = gtpet

                optimizer_trans.zero_grad()

                fake_pet = model(inpmri)
                loss = F.mse_loss(fake_pet, GT)

                loss.backward()
                loss_save += loss
                optimizer_trans.step()
                train_loader_with_progress.set_postfix(loss=f"{loss:.4f}")

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            n = 0
            for (inpmri, gtpet, MCimages, xiaobo_pet, wenli) in val_loader_with_progress:
                model.eval()
                with torch.no_grad():
                    inpmri, gtpet, MCimages, xiaobo_pet, wenli = inpmri.to(device), gtpet.to(
                        device), MCimages.to(device), xiaobo_pet.to(device), wenli.to(device)
                    out = model(inpmri)
                    gtpet_numpy = wenli.clone().cpu().numpy()
                    out_numpy = out.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                    nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                    nmse_all += nmse
                    nmae_all += nmae

                    ssim_all += ssim_scale
                    psnr_all += psnr_scale
                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}")

                    save_tensor_to_nii(out, 'dvae.nii')
                    save_tensor_to_nii(gtpet, 'gtpet.nii')
                    save_tensor_to_nii(xiaobo_pet, 'xiaobo_pet.nii')
                    n += 1

            print(psnr_all / 21, ssim_all / 21)

        len_val = len(val_dataloader)

        if max_psnr < psnr_all:
            max_psnr = psnr_all
            torch.save(model.state_dict(), translate_model_save_path)
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if max_ssim < ssim_all:
            max_ssim = ssim_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")

        if nmse_all < min_nmse:
            min_nmse = nmse_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if nmae_all < min_nmae:
            min_nmae = nmae_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if epoch == epochs - 1:
            print(
                f"Final model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")





import torch
import torch.nn as nn
import torch.nn.functional as F



class PatchL1Loss3D(nn.Module):
    def __init__(self, patch_size=16, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, fake, real):
        B, C, D, H, W = fake.shape

        # 计算能提取多少个 patch（避免越界）
        D_out = (D - self.patch_size) // self.stride + 1
        H_out = (H - self.patch_size) // self.stride + 1
        W_out = (W - self.patch_size) // self.stride + 1

        if D_out <= 0 or H_out <= 0 or W_out <= 0:
            return F.l1_loss(fake, real)  # 回退到全局 L1

        # 用 unfold 展开成 patches
        # fake: [B, C, D, H, W] → 先 reshape 成 [B*C, 1, D, H, W] 以便 unfold
        fake_reshaped = fake.view(B * C, 1, D, H, W)
        real_reshaped = real.view(B * C, 1, D, H, W)

        # 使用 unfold 沿每个空间维度展开
        # 输出: [B*C, 1 * ps*ps*ps, N_patches]
        fake_patches = fake_reshaped.unfold(2, self.patch_size, self.stride) \
                                  .unfold(3, self.patch_size, self.stride) \
                                  .unfold(4, self.patch_size, self.stride)
        real_patches = real_reshaped.unfold(2, self.patch_size, self.stride) \
                                  .unfold(3, self.patch_size, self.stride) \
                                  .unfold(4, self.patch_size, self.stride)

        # 合并最后三个维度 (ps, ps, ps) → [B*C, ps³, N]
        ps = self.patch_size
        N = fake_patches.size(-1) * fake_patches.size(-2) * fake_patches.size(-3)
        fake_patches = fake_patches.contiguous().view(B * C, ps*ps*ps, -1)
        real_patches = real_patches.contiguous().view(B * C, ps*ps*ps, -1)

        # 计算每个 patch 的 L1 Loss
        # [B*C, ps³, N] → mean over ps³ → [B*C, N] → mean over N
        l1_per_patch = torch.mean(torch.abs(fake_patches - real_patches), dim=1)  # [B*C, N]
        loss = torch.mean(l1_per_patch)  # scalar

        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_histogram_3d(x, bins=256, min_val=0.0, max_val=1.0):
    """
    3D 可微分软直方图（支持 [N, C, D, H, W] 或 [N, D, H, W]）
    x: 输入张量，值应在 [min_val, max_val] 范围内
    返回: [N, C, bins] 或 [N, bins] —— 每个样本每个通道的归一化分布
    """
    if x.dim() == 5:  # [N, C, D, H, W]
        N, C, D, H, W = x.shape
        x = x.view(N, C, -1)  # [N, C, D*H*W]
    elif x.dim() == 4:  # [N, D, H, W]
        N, D, H, W = x.shape
        x = x.view(N, -1)     # [N, D*H*W]
    else:
        raise ValueError("Input must be 4D (N, D, H, W) or 5D (N, C, D, H, W)")

    delta = (max_val - min_val) / bins
    centers = torch.linspace(min_val + delta / 2, max_val - delta / 2, bins, device=x.device)
    centers = centers.view(1, 1, -1)  # [1, 1, bins]

    if x.dim() == 3:  # 多通道情况 [N, C, L]
        x = x.unsqueeze(3)  # [N, C, L, 1]
        centers = centers.unsqueeze(0)  # [1, 1, 1, bins] → 广播成 [N, C, L, bins]
        dist = torch.abs(x - centers)   # [N, C, L, bins]
        hist = torch.clamp(1 - dist / delta, min=0)  # 三角隶属函数
        hist = hist.sum(dim=2)          # [N, C, bins]
        hist = hist / hist.sum(dim=2, keepdim=True).clamp(min=1e-8)  # 归一化
        return hist  # [N, C, bins]
    else:  # 单通道/灰度 [N, L]
        x = x.unsqueeze(2)  # [N, L, 1]
        dist = torch.abs(x - centers)   # [N, L, bins]
        hist = torch.clamp(1 - dist / delta, min=0)
        hist = hist.sum(dim=1)          # [N, bins]
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return hist  # [N, bins]

def wasserstein_distance_1d(p, q):
    """
    一维Wasserstein-1距离（适用于直方图）
    p, q: [N, bins] 或 [N, C, bins]
    """
    if p.dim() == 3:  # 多通道 [N, C, bins]
        cdf_p = torch.cumsum(p, dim=2)
        cdf_q = torch.cumsum(q, dim=2)
        return torch.mean(torch.abs(cdf_p - cdf_q), dim=2).mean()  # 标量 loss
    else:  # [N, bins]
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)
        return torch.mean(torch.abs(cdf_p - cdf_q))



class DistributionMatchingLoss3D(nn.Module):
    def __init__(self, bins=128, min_val=0.0, max_val=1.0, loss_type='wasserstein'):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.loss_type = loss_type

    def forward(self, fake, real):
        """
        fake, real: 3D 张量，形状为 [N, C, D, H, W] 或 [N, D, H, W]
        值范围应在 [min_val, max_val]
        """
        if fake.dim() not in [4, 5] or real.dim() not in [4, 5]:
            raise ValueError("Input must be 4D or 5D tensor for 3D volumes.")

        if fake.dim() == 5:  # 多通道处理
            loss = 0.0
            for c in range(fake.size(1)):
                hist_fake = soft_histogram_3d(fake[:, c], self.bins, self.min_val, self.max_val)
                hist_real = soft_histogram_3d(real[:, c], self.bins, self.min_val, self.max_val)

                if self.loss_type == 'wasserstein':
                    loss += wasserstein_distance_1d(hist_fake, hist_real)
                elif self.loss_type == 'kl':
                    loss += F.kl_div(hist_fake.log(), hist_real, reduction='batchmean')
                elif self.loss_type == 'mse':
                    loss += F.mse_loss(hist_fake, hist_real)
            return loss / fake.size(1)
        else:  # 单通道
            hist_fake = soft_histogram_3d(fake, self.bins, self.min_val, self.max_val)
            hist_real = soft_histogram_3d(real, self.bins, self.min_val, self.max_val)

            if self.loss_type == 'wasserstein':
                return wasserstein_distance_1d(hist_fake, hist_real)
            elif self.loss_type == 'kl':
                return F.kl_div(hist_fake.log(), hist_real, reduction='batchmean')
            elif self.loss_type == 'mse':
                return F.mse_loss(hist_fake, hist_real)
def replace_slices(tensor1, tensor2, replacement_ratio):

    D = tensor1.size(2)
    num_replace = int(D * replacement_ratio)

    if num_replace == 0:
        result = tensor1.clone()
    else:
        replace_indices = torch.randperm(D)[:num_replace]
        result = tensor1.clone()
        result[:, :, replace_indices, :, :] = tensor2[:, :, replace_indices, :, :]

    return result




import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import pad as torch_pad

# Radon 变换函数（适用于 2D 切片）
def radon_torch_2d(
    img: torch.Tensor,
    angles: torch.Tensor,
    degrees: bool = True,
    circle: bool = False,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
):
    """
    2D Radon 变换函数
    """
    N, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # 角度处理
    angles = angles.to(device=device, dtype=dtype)
    if degrees:
        angles_rad = angles * (torch.pi / 180.0)
    else:
        angles_rad = angles

    # 自动计算 padding，使得图像不被裁剪
    L = int(np.ceil(np.sqrt(H * H + W * W)))  # 最大的探测器长度
    pad_h = max(0, L - H)
    pad_w = max(0, L - W)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # 改名为 padding
    img_pad = torch_pad(img, pad=padding, mode="constant", value=0.0)

    # 圆形视野（可选）
    if circle:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, L, device=device, dtype=dtype),
            torch.linspace(-1, 1, L, device=device, dtype=dtype),
            indexing="ij",
        )
        r = min(H, W) / float(L)
        mask = ((xx**2 + yy**2) <= r**2).to(dtype=dtype)
        img_pad = img_pad * mask.view(1, 1, L, L)

    # 计算旋转矩阵并应用
    A = angles_rad.numel()
    sinogram = []

    cos_t = torch.cos(angles_rad)
    sin_t = torch.sin(angles_rad)

    thetas = torch.stack([
        torch.stack([cos_t, -sin_t, torch.zeros_like(cos_t)], dim=-1),
        torch.stack([sin_t, cos_t, torch.zeros_like(sin_t)], dim=-1),
    ], dim=1)

    x = img_pad.view(1 * C, 1, L, L)  # Batch expand

    x_rep = x.repeat(A, 1, 1, 1)
    thetas_rep = thetas.repeat_interleave(1 * C, dim=0)

    grid = torch.nn.functional.affine_grid(thetas_rep, size=x_rep.shape, align_corners=align_corners)
    x_rot = torch.nn.functional.grid_sample(x_rep, grid, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)

    proj = x_rot.sum(dim=2)

    proj = proj.view(A, 1 * C, 1, L).permute(1, 2, 0, 3)
    sinogram = proj.view(1, C, A, L)

    return sinogram

# 读取 3D PET 数据并进行 Radon 变换
def radon_transform_3d_pet(pet_tensor: str, angles=torch.linspace(0, 120, steps=120) , degrees: bool = True, circle: bool = False, axis: int = 2):

    # 获取 3D 图像的形状
    #pet_tensor = pet_tensor.squeeze()
    _,_,D, H, W = pet_tensor.shape

    # 初始化 sinogram
    full_sinogram = []

    # 对选择的轴（维度）进行切片
    if axis == 0:  # x轴方向
        for d in range(D):
            slice_2d = pet_tensor[:, :, d, :]  # 获取该切片 (1,1,H,W)
            sinogram = radon_torch_2d(slice_2d, angles, degrees=degrees, circle=circle)
            full_sinogram.append(sinogram)
    elif axis == 1:  # y轴方向
        for d in range(H):
            slice_2d = pet_tensor[:, :, :, d]  # 获取该切片 (1,1,D,W)
            sinogram = radon_torch_2d(slice_2d, angles, degrees=degrees, circle=circle)
            full_sinogram.append(sinogram)
    elif axis == 2:  # z轴方向
        for d in range(W):
            slice_2d = pet_tensor[:, :, :, d]  # 获取该切片 (1,1,D,H)
            sinogram = radon_torch_2d(slice_2d, angles, degrees=degrees, circle=circle)
            full_sinogram.append(sinogram)
    else:
        raise ValueError("Invalid axis value. Choose from 0 (x), 1 (y), or 2 (z).")

    # 将 sinogram 组合成一个完整的 sinogram
    full_sinogram = torch.cat(full_sinogram, dim=0)  # (D, C, A, L)
    return full_sinogram

class Radon3D(torch.nn.Module):
    def __init__(self, ax=2):
        super().__init__()
        self.ax = ax

    def forward(self, out, gt):
        # 对输出和 ground truth 执行 Radon 变换
        out = radon_transform_3d_pet(out)
        gt = radon_transform_3d_pet(gt)

        # 初始损失计算
        loss = F.l1_loss(out[0], gt[0])

        # 遍历并累加每个 sinogram 的损失
        for i in range(1, len(out)):  # 不需要 i = i + 1，直接从 1 开始
            loss += F.l1_loss(out[i], gt[i])

        return loss



def main(train_dataloader, val_dataloader, epochs, device, resume, model_save_path=None,
         only_eval=False):

    if model_save_path is None:
        print("No model saved")


    model_save_path = model_save_path
    model = PlainConvUNet(1, 6, (32, 64, 128, 256,320,320), nn.Conv3d, 3, (1, 2, 2, 2,2,2), (2, 2, 2, 2,2,2), 1,
                          (2, 2, 2,2,2),False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0).to(
        device)
    



    if resume:
        model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)

    combined_parameters = list(model.parameters()) 
    optimizer = torch.optim.AdamW(combined_parameters, lr=0.00001)


    max_psnr = 0
    max_ssim = 0
    min_nmse = float('inf')
    min_nmae = float('inf')

    pl1 = PatchL1Loss3D(patch_size=2, stride=1).to(device)

    for epoch in range(epochs):

        alpha = 0
        
        if not only_eval:
            loss_save = 0
            model.train()
            train_loader_with_progress = tqdm(train_dataloader,
                                              desc=f'Epoch [{epoch + 1}/{epochs}]',
                                              leave=False)  # leave=False 表示结束后清除进度条
            for (inpmri, gtpet, MCimages, mri_wenli, name,ceb) in train_loader_with_progress:  
                inpmri, gtpet, MCimages, mri_wenli = inpmri.to(device), gtpet.to(device), MCimages.to(
                    device),  mri_wenli.to(device)
                MCimages = MCimages.view(MCimages.size(0), MCimages.size(1), -1, MCimages.size(4), MCimages.size(5))
                if ceb is not None:
                    ceb = ceb.to(device)
                optimizer.zero_grad()
                mask = (gtpet > 0).float() 

                pet_out = model(inpmri, MCimages,mri_wenli,ceb)

                output = pet_out

                pet_out = pet_out * mask

                # pre = model_D(inpmri,pet_out)
                # fake_loss = F.mse_loss(pre,torch.ones_like(pre))

                loss =  pl1(pet_out,gtpet)*10 + F.l1_loss(pet_out,gtpet)*50# + G_loss(pet_out,gtpet)*10


                loss.backward()
                loss_save += loss
                optimizer.step()

                train_loader_with_progress.set_postfix(loss=f"{loss.mean().item():.4f}")

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            for (inpmri, gtpet, MCimages, mri_wenli, name,ceb) in val_loader_with_progress:
                #model.eval()
                with torch.no_grad():
                    #inpmri = replace_slices(inpmri, gtpet, alpha)
                    inpmri, gtpet, MCimages,mri_wenli = inpmri.to(device), gtpet.to(
                        device), MCimages.to(device),  mri_wenli.to(device)
                    if ceb is not None:
                        ceb = ceb.to(device)
                    MCimages = MCimages.view(MCimages.size(0), MCimages.size(1), -1, MCimages.size(3), MCimages.size(4))
                    mask = (gtpet > 0).float() 
                    # fake_pet = model2(inpmri)
                    # inpmri_out = model2(inpmri, MCimages,mri_wenli)
                    # inpmri_out = inpmri_out * mask
                    pet_out = model(inpmri, MCimages, mri_wenli,ceb)

                    # pet_out, wenli_out = torch.split(pet_out, 1, dim=1)
                    output = pet_out

                     
                    output = output * mask  

                    gtpet_numpy = gtpet.clone().cpu().numpy()
                    out_numpy = output.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                    nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                    nmse_all += nmse
                    nmae_all += nmae

                    ssim_all += ssim_scale
                    psnr_all += psnr_scale
                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}")

                    # if psnr_scale >= 30:

                    dir_name = 'pl'
                    gtname = dir_name + 'gt_' + name[0]
                    outname = name[0]
                    mri_name = dir_name + 'mri_' + name[0]
                    mri_out = dir_name + 'mriout_' + name[0]

                    save_tensor_to_nii(output, outname)
                    # save_tensor_to_nii(inpmri, mri_name)
                    # save_tensor_to_nii(gtpet, gtname)
                    # save_tensor_to_nii(mri_wenli,mri_out)
                    #save_tensor_to_nii(inpmri_out, mri_out)
                    # save_tensor_to_nii(wenli_out, 'CIwenliout.nii')
                # save_tensor_to_nii(fake_pet, 'ADNIfakepet.nii')

            print(psnr_all / 65, ssim_all / 65)

        len_val = len(val_dataloader)

        if max_psnr < psnr_all:
            max_psnr = psnr_all
            torch.save(model.state_dict(), model_save_path)
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if max_ssim < ssim_all:
            max_ssim = ssim_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")

        if nmse_all < min_nmse:
            min_nmse = nmse_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if nmae_all < min_nmae:
            min_nmae = nmae_all
            print(
                f"New best model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")
        if epoch == epochs - 1:
            print(
                f"Final model saved with PSNR: {max_psnr / len_val:.4f}, SSIM: {max_ssim / len_val:.4f} ,NMSE: {min_nmse / len_val:.4f}, NMAE: {min_nmae / len_val:.4f}")

