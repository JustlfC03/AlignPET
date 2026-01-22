
import os
import sys

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from model.FuseUnet.Decoder import MyDecoder

import pywt
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from natsort import decoder

from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from generative.losses import PerceptualLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from typing import Union, Type, List, Tuple

from model.HLVAE.model import HLVAE
from model.VAR.models.VAEtry import VAE
from evaluation.eval import save_tensor_to_nii,calculate_nmse_nmae
# from dataloader.dataloader3D import myDataset3D
# from dataloader.dataloader2D import myDataset3

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

def xiaobo(PET_tensor, MRI_tensor):
    # PET_tensor.cpu().numpy()
    # MRI_tensor.cpu().numpy()

    original_device = PET_tensor.device

    PET_tensor = PET_tensor.squeeze()  # 去掉batch维度\
    MRI_tensor = MRI_tensor.squeeze()  # 去掉batch维度

    PET_tensor = PET_tensor.detach().cpu().numpy()
    MRI_tensor = MRI_tensor.detach().cpu().numpy()
    # 小波变换
    GT_coeffs = pywt.dwtn(PET_tensor, wavelet='haar', mode='symmetric')
    OUT_coeffs = pywt.dwtn(MRI_tensor, wavelet='haar', mode='symmetric')

    change = [0, 1, 2, 3, 4, 5, 6, 7]
    weight = 0  # .3
    for comp_idx, comp_key in enumerate(GT_coeffs.keys()):
        if comp_idx in change:
            if comp_idx == 0:
                GT_coeffs[comp_key] = OUT_coeffs[comp_key] * weight + GT_coeffs[comp_key] * (1 - weight)
            else:
                GT_coeffs[comp_key] = OUT_coeffs[comp_key]

    reconstructed_GT = pywt.idwtn(GT_coeffs, wavelet='haar', mode='symmetric')
    # reconstructed_OUT = pywt.idwtn(OUT_coeffs, wavelet='haar', mode='symmetric')

    reconstructed_GT = (reconstructed_GT - np.min(reconstructed_GT)) / (
                np.max(reconstructed_GT) - np.min(reconstructed_GT))  # 归一化到0-1范围
    reconstructed_GT = torch.from_numpy(reconstructed_GT).unsqueeze(0).unsqueeze(0).to(original_device)  # 恢复batch维度

    # print(f"reconstructed_GT shape: {reconstructed_GT.shape}, GT_tensor shape: {PET_tensor.shape}")

    return reconstructed_GT


class CrossAttentionFusion(nn.Module):
    def __init__(self, stage_channels, guidance_channels=96, d_model=64, reduction_ratio=4):
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

    def forward(self, x, var_condition):
        ret = []
        for i in range(len(self.stages)):
            s = self.stages[i]
            v = self.var_condition[i]
            x = s(x)
            x = v(x, var_condition)

            # print("encoder",x.shape)
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

    def forward(self, x, var_condition):
        skips = self.encoder(x, var_condition)
        if mod:
            return self.decoder(x, skips)
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



class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
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
        init_last_bn_before_add_to_0(module)


def train_trans(train_dataloader, val_dataloader, epochs, device, resume,HL_VAE_path=None,not_xiaobo = False,only_eval=False):
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
            for (inpmri, gtpet, MCimages, xiaobo_pet) in train_loader_with_progress:
                inpmri,  gtpet, MCimages, xiaobo_pet = inpmri.to(device), gtpet.to(
                    device), MCimages.to(device), xiaobo_pet.to(device)

                GT = xiaobo_pet
                if not_xiaobo:
                    GT = gtpet

                optimizer_trans.zero_grad()

                fake_pet = model(inpmri)
                loss = F.l1_loss(fake_pet, GT)

                loss.backward()
                loss_save += loss
                optimizer_trans.step()
                train_loader_with_progress.set_postfix(loss=f"{loss:.4f}")

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        len_val = len(val_dataloader)

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            n = 0
            for (inpmri, gtpet, MCimages, xiaobo_pet) in val_loader_with_progress:
                model.eval()
                with torch.no_grad():
                    inpmri,  gtpet, MCimages, xiaobo_pet = inpmri.to(device),  gtpet.to(
                        device), MCimages.to(device), xiaobo_pet.to(device)
                    out = model(inpmri)
                    GT = xiaobo_pet
                    if not_xiaobo:
                        GT = gtpet.clone()
                    gtpet_numpy = GT.clone().cpu().numpy()
                    out_numpy = out.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                    nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                    nmse_all += nmse
                    nmae_all += nmae

                    ssim_all += ssim_scale
                    psnr_all += psnr_scale
                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}")


                    save_tensor_to_nii(out, 'xuanwudvaeX.nii')
                    save_tensor_to_nii(gtpet, 'xuanwugtpetX.nii')
                    save_tensor_to_nii(xiaobo_pet, 'xuanwuxiaobo_petX.nii')
                    n += 1

            print(psnr_all / len_val, ssim_all / len_val)

        

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


def main(train_dataloader, val_dataloader, epochs, device, resume, model_in='mri',model_save_path=None,HL_VAE_path=None,step = 0,not_xiaobo = False,only_eval=False):
    translate_model = HLVAE().to(device)

    if model_save_path is None:
        print("No model saved")
    if HL_VAE_path is None:
        print("No HL_VAE model saved")

    model = PlainConvUNet(1, 4, (32, 64, 128, 256), nn.Conv3d, 3, (1, 2, 2, 2), (2, 2, 2, 2), 1,
                          (2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0).to(
        device)

    model_save_path = model_save_path
    model2_save_path =HL_VAE_path
    if resume:
        model.load_state_dict(torch.load(model_save_path))
    translate_model.load_state_dict(torch.load(model2_save_path))

    model = model.to(device)
    model2 = translate_model.to(device)
    if step == 0:
        combined_parameters = list(model.parameters())
    if step == 1:
        combined_parameters = list(model.parameters()) + list(model2.parameters())
    if step != 0 and step != 1:
        print("The stage is illegal.")

    optimizer = torch.optim.AdamW(combined_parameters, lr=0.0001)

    max_psnr = 0
    max_ssim = 0
    min_nmse = float('inf')
    min_nmae = float('inf')

    for epoch in range(epochs):
        if not only_eval:
            loss_save = 0
            model.train()
            model2.train()
            train_loader_with_progress = tqdm(train_dataloader,
                                              desc=f'Epoch [{epoch + 1}/{epochs}]',
                                              leave=False)  # leave=False 表示结束后清除进度条
            for (inpmri, gtpet, MCimages, xiaobo_pet) in train_loader_with_progress:
                # batch = batch.to(device)
                inpmri, gtpet, MCimages, xiaobo_pet = inpmri.to(device), gtpet.to(device), MCimages.to(device), xiaobo_pet.to(device)
                MCimages = MCimages.view(MCimages.size(0), MCimages.size(1), -1, MCimages.size(4), MCimages.size(5))
                optimizer.zero_grad()

                if model_in == 'mri':
                    fake_pet = model2(inpmri)
                    output = model(fake_pet, MCimages)
                if model_in == 'pet':
                    output = model(xiaobo_pet, MCimages)
                GT = xiaobo_pet
                if not_xiaobo:
                    GT = gtpet

                loss = F.l1_loss(output, gtpet)
                if step == 1:
                    loss = loss + F.mse_loss(fake_pet, GT) * 0.5

                loss.backward()
                loss_save += loss
                optimizer.step()
                train_loader_with_progress.set_postfix(loss=f"{loss:.4f}")

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            for (inpmri, labelB, gtpet, MCimages, xiaobo_pet) in val_loader_with_progress:
                model.eval()
                model2.eval()
                with torch.no_grad():
                    inpmri, labelB, gtpet, MCimages, xiaobo_pet = inpmri.to(device), labelB.to(device), gtpet.to(
                        device), MCimages.to(device), xiaobo_pet.to(device)
                    MCimages = MCimages.view(MCimages.size(0), MCimages.size(1), -1, MCimages.size(3), MCimages.size(4))
                    if model_in == 'mri':
                        fake_pet = model2(inpmri)
                        output = model(fake_pet, MCimages)
                    if model_in == 'pet':
                        output = model(xiaobo_pet, MCimages)

                    mask = (gtpet > 0.05).float()  # 创建一个mask，非零部分为1，零部分为0
                    # out = out * mask  # 将输出和mask相乘，保留非零部分
                    # gtpet = gtpet * mask  # 将ground truth和mask相乘，保留非零部分

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
                    save_tensor_to_nii(output, 'ADNI.nii')
                    save_tensor_to_nii(gtpet, 'ADNIgtpet.nii')


            print(psnr_all / 21, ssim_all / 21)

        len_val = len(val_dataloader)

        if max_psnr < psnr_all:
            max_psnr = psnr_all
            if epoch != 0:
                torch.save(model.state_dict(), model_save_path)
                torch.save(translate_model.state_dict(), model2_save_path)
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


def train_VAE(train_dataloader, val_dataloader, epochs, device, optimizer=None,model_save_path = None):

    model = VAE()
    

    state_dict = torch.load(model_save_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    loss_max = 10000000000
    for epoch in range(epochs):
        loss_save = 0
        model.train()
        train_loader_with_progress = tqdm(train_dataloader,
                                          desc=f'Epoch [{epoch + 1}/{epochs}]',
                                          leave=False)  # leave=False 表示结束后清除进度条
        for (inpmri, gtpet) in train_loader_with_progress:
            # batch = batch.to(device)
            inpmri, gtpet = inpmri.to(device), gtpet.to(device)
            optimizer.zero_grad()
            output = model(inpmri)
            loss = F.mse_loss(output, inpmri)  # ,mu,logvar)
            #loss += F.mse_loss(mri_latent, pet_latent)
            loss.backward()
            loss_save += loss
            optimizer.step()
            train_loader_with_progress.set_postfix(loss=f"{loss:.4f}")
        if loss_save < loss_max:
            loss_max = loss_save
            print(loss_max)
            torch.save(model.state_dict(), model_save_path)
        if epoch % 1 == 0:

            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            for (inpmri,  gtpet,fn) in val_loader_with_progress:
                model.eval()
                with torch.no_grad():
                    inpmri,gtpet = inpmri.to(device), gtpet.to(device)
                    inpmri,gtpet = inpmri.squeeze(0),gtpet.squeeze(0)
                    out= model(inpmri)
                    save_tensor_to_nii(out, 'xuanwuVAEOUT.nii')

                    out = out[:, 0:1, :, :]
                    gtpet = inpmri[:, 0:1, :, :]

                    gtpet_numpy = gtpet.clone().cpu().numpy()
                    out_numpy = out.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=1, data_range=1)

                    #nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)

                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}", ssim=f"{ssim_scale:.4f}")
                    break
            print(psnr_scale, ssim_scale)


