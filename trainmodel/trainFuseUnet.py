from typing import Union, Type, List, Tuple
from torch.utils.data import DataLoader, random_split
import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.nets.unet_decoder import MyDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import pdb
import os
from dataloader.dataloader_compara import myDataset3D
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import os

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.functional as F1
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_nmse_nmae(gt, recon):
    '''ADNI数据集的最后一层和第一层存在异常值，去除'''
    gt = gt.squeeze()[1:-1, ...]
    recon = recon.squeeze()[1:-1, ...]

    y = np.asarray(gt, dtype=np.float64)
    y_hat = np.asarray(recon, dtype=np.float64)

    if y.shape != y_hat.shape:
        raise ValueError(f"形状不匹配: gt {y.shape} vs recon {y_hat.shape}")

    # 展平数组以便计算
    y_flat = y.ravel()
    y_hat_flat = y_hat.ravel()

    # 计算NMSE
    numerator_nmse = np.sum((y_flat - y_hat_flat) ** 2)
    denominator_nmse = np.sum(y_flat ** 2)

    # 避免除以零
    if denominator_nmse == 0:
        nmse = 0.0 if numerator_nmse == 0 else np.inf
    else:
        nmse = numerator_nmse / denominator_nmse

    # 计算NMAE
    numerator_nmae = np.sum(np.abs(y_flat - y_hat_flat))
    denominator_nmae = np.sum(np.abs(y_flat))

    # 避免除以零
    if denominator_nmae == 0:
        nmae = 0.0 if numerator_nmae == 0 else np.inf
    else:
        nmae = numerator_nmae / denominator_nmae

    return nmse, nmae


def save_tensor_to_nii(tensor, name):
    base_path = '/home/cyf/CAPL/VAR-mainbase/VARrecon/mid3'
    os.makedirs(base_path, exist_ok=True)

    data = tensor.squeeze().cpu().numpy()

    single_data = data.squeeze()
    single_data = np.transpose(single_data, [1, 2, 0])
    img = nib.Nifti1Image(single_data, affine=np.eye(4))
    save_path = os.path.join(base_path, name)
    nib.save(img, save_path)


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
                 nonlin_first: bool = False,
                 device=None
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
                                     nonlin_first=nonlin_first).to(device)
        else:
            self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                       nonlin_first=nonlin_first).to(device)

    def forward(self, x):
        skips = self.encoder(x)
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


class ResidualEncoderUNet(nn.Module):
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
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

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


def main(train_dataloader, val_dataloader, epochs, device, resume,model_save_path):
    model = PlainConvUNet(1, 6, (32, 64, 128, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
                          (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=0).to(
        device)
    if resume:
        model.load_state_dict(torch.load(model_save_path))

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)

    max_psnr = 0
    max_ssim = 0
    min_nmse = float('inf')
    min_nmae = float('inf')

    for epoch in range(epochs):
        loss_save = 0
        loss_max = 1000000000
        model.train()
        train_loader_with_progress = tqdm(train_dataloader,
                                          desc=f'Epoch [{epoch + 1}/{epochs}]',
                                          leave=False)  # leave=False 表示结束后清除进度条
        for (inpmri, gtpet) in train_loader_with_progress:
            # batch = batch.to(device)
            inpmri, gtpet = inpmri.to(device), gtpet.to(device)

            optimizer.zero_grad()

            out = model(inpmri)

            loss = F.mse_loss(out, gtpet)  # ,mu,logvar)

            loss.backward()
            loss_save += loss
            optimizer.step()
            train_loader_with_progress.set_postfix(loss=f"{loss:.4f}")

        psnr_all = 0
        ssim_all = 0
        nmse_all = 0
        nmae_all = 0

        num = 0

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            for (inpmri, gtpet) in val_loader_with_progress:
                # if num != 1:
                #     num += 1
                #     continue
                # num += 1
                model.eval()
                with torch.no_grad():
                    inpmri, gtpet = inpmri.to(device), gtpet.to(
                        device)
                    out = model(inpmri)

                    # mask = (gtpet > 0.05).float()  # 创建一个mask，非零部分为1，零部分为0
                    # out = out * mask  # 将输出和mask相乘，保留非零部分
                    # gtpet = gtpet * mask  # 将ground truth和mask相乘，保留非零部分

                    gtpet_numpy = gtpet.clone().cpu().numpy()
                    out_numpy = out.clone().cpu().numpy()
                    psnr_scale = psnr(gtpet_numpy, out_numpy)
                    ssim_scale = ssim(gtpet_numpy.squeeze(), out_numpy.squeeze(), channel_axis=-1, data_range=1)

                    nmse, nmae = calculate_nmse_nmae(gtpet_numpy, out_numpy)
                    nmse_all += nmse
                    nmae_all += nmae

                    ssim_all += ssim_scale
                    psnr_all += psnr_scale
                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}")

                    if psnr_scale >= 30:
                        save_tensor_to_nii(out, 'dvae.nii')
                        save_tensor_to_nii(gtpet, 'gtpet.nii')
                        # save_tensor_to_nii(gtpet-out+(mask), 'xuanwubiaspetmask.nii')

            print(psnr_all, ssim_all)

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


