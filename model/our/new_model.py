import time
from typing import List, Optional, Tuple, Union

import torch
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
import nibabel as nib
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
from torch.utils.tensorboard import SummaryWriter
import os

from model.our.mynew import VAE


Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

import math
from typing import List, Optional, Tuple, Union

import torch

def save_tensor_to_nii(tensor, name):
    # 设置保存路径的基目录
    base_path = ''

    # 确保目录存在
    os.makedirs(base_path, exist_ok=True)

    # 将 tensor 移动到 CPU 并转为 numpy
    data = tensor.squeeze().cpu().numpy()

    datas = []
    # 检查 batch 维度是否存在
    if tensor.shape[0] == 1:
        # 单个样本的情况
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_path, name)
        nib.save(img, save_path)
    else:

        data = tensor[:, 1, :, :].cpu().numpy()
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_path, name)
        nib.save(img, save_path)


def calculate_psnr(gt: torch.Tensor, recon: torch.Tensor, max_val: float = 1.0):
    assert gt.shape == recon.shape, "gt 和 recon 的形状必须相同"

    mse = torch.mean((gt - recon) ** 2, dim=[1, 2, 3])  # 每张图的 MSE
    psnr = 10 * torch.log10(max_val ** 2 / mse)  # 每张图的 PSNR
    return psnr.mean().item()


def cnmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)  # 使用 torch.mean
    mean_y = torch.mean(y_true)
    nmse_value = mse / (mean_y ** 2)
    return nmse_value.item()  # 如果你想返回一个 Python float



def main_train(train_dataloader, val_dataloader, epochs, device, resume, model_save_path=None):

    model = VAE().to(device)

    max_psnr_ssim = 0.0

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=0.0001)

    path = ''
    model.load_state_dict(torch.load(path))
    model.to(device)


    for epoch in range(epochs):
        # ---------- Training phase ----------
        a = 1
        if epoch>=10:
            a=2

        train_loader_with_progress = tqdm(train_dataloader,
                                          desc=f'Epoch [{epoch + 1}/{epochs}]',
                                          leave=False)

        total_train_loss = 0.0
        num_train_batches = 0
        model.train()
        for (inpmri, gtpet, fn) in train_loader_with_progress:
            optimizer.zero_grad()
            inpmri = inpmri.squeeze()
            gtpet = gtpet.squeeze()
            inpmri = inpmri.to(device)
            gtpet = gtpet.to(device)

            if a == 1:
                inpmri = gtpet
            if a == 2:
                gtpet =inpmri

            out = model(inpmri)
            loss = F.l1_loss(out,gtpet)

            total_train_loss += loss.item()
            num_train_batches += 1
            loss.backward()
            optimizer.step()

            train_loader_with_progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        print(avg_train_loss)

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            model.eval()

            total_psnr = 0.0
            total_nmse = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for (inpmri, gtpet, fn) in val_loader_with_progress:
                    inpmri, gtpet = inpmri.to(device), gtpet.to(device)


                    if a == 1:
                        inpmri = gtpet
                    if a == 2:
                        gtpet =inpmri

                    out = model(inpmri)
                    psnr_scale = calculate_psnr(gtpet, out)
                    total_psnr += psnr_scale
                    num_val_batches += 1
                    save_tensor_to_nii(out, 'mutiscaleOUT.nii')
                    save_tensor_to_nii(gtpet, 'mutiscaleGT.nii')
                    val_loader_with_progress.set_postfix(
                        psnr=f"{psnr_scale:.4f}"
                    )
            avg_psnr = total_psnr / num_val_batches


            new_model_save_path = model_save_path.replace('.pth', '') + '_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), new_model_save_path)

            print(f"Epoch {epoch + 1} - Val: PSNR={avg_psnr:.4f}")

