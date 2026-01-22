import os
import sys

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from model.VAR.models.var import VQVAE
from torch.utils.data import DataLoader



from pprint import pformat

import math
from typing import List, Optional, Tuple, Union

import torch
import nibabel as nib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from evaluation.eval import save_tensor_to_nii

def calculate_psnr(gt: torch.Tensor, recon: torch.Tensor, max_val: float = 1.0):
    assert gt.shape == recon.shape, "gt 和 recon 的形状必须相同"
    mse = torch.mean((gt - recon) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return psnr.mean().item()



def ssim(img1, img2, window_size=7, size_average=True, full=False):  # 有点问题
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    window = torch.ones(3, 1, window_size, window_size).float().to(device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=3)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        ssim_value = torch.mean(ssim_map)
    else:
        ssim_value = torch.mean(ssim_map, dim=(1, 2, 3))

    if full:
        return ssim_value, ssim_map
    else:
        return ssim_value

class NullCtx:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
            self,
            model_maybe_fsdp: Union[torch.nn.Module, FSDP], fp16: bool, bf16: bool, zero: int,
            optimizer: torch.optim.Optimizer, grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.model_maybe_fsdp = model_maybe_fsdp
        self.zero = zero
        self.enable_amp = fp16 or bf16
        self.using_fp16_rather_bf16 = fp16

        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True,
                                          dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16,
                                          cache_enabled=self.zero == 0)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11,
                                                    growth_interval=1000) if self.using_fp16_rather_bf16 else None  # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None

        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')

        self.r_accu = 1 / n_gradient_accumulation  # r_accu == 1.0 / n_gradient_accumulation

    def backward_clip_step(
            self, stepping: bool, loss: torch.Tensor,
    ):
        # backward
        loss = loss.mul(self.r_accu)  # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)
            if self.early_clipping:
                if self.zero:
                    orig_norm: Optional[torch.Tensor] = self.model_maybe_fsdp.clip_grad_norm_(self.grad_clip)
                else:
                    orig_norm: Optional[torch.Tensor] = torch.nn.utils.clip_grad_norm_(
                        self.model_maybe_fsdp.parameters(), self.grad_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: Optional[float] = self.scaler.get_scale()
                if scaler_sc > 65536.:  # fp16 will overflow when >65536, so multiply 65536 could be dangerous
                    self.scaler.update(new_scale=65536.)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()

            if self.late_clipping:
                orig_norm: Optional[torch.Tensor] = self.optimizer.global_grad_norm

            self.optimizer.zero_grad(set_to_none=True)

        return orig_norm, scaler_sc, loss

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try:
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e:
                print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])


def filter_params(model, nowd_keys=()):
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)

        if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'
        cur_lr_sc = 1.
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)

    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)

    # print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')

    assert len(
        names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return names, paras, list(para_groups.values())


def main_train(train_dataloader, val_dataloader, epochs,device,vae_paras_path=None,vae_save_path=None,only_eval=False):
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4

    ac = 1
    fp16: int = 0
    tclip: float = 2.
    model = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                  v_patch_nums=patch_nums).to(device)
    if vae_paras_path is not None:
        vae_state_dict = torch.load(vae_paras_path, map_location=device)
        model.load_state_dict(vae_state_dict, strict=True)
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = True

    nowd_keys = {
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'class_emb', 'embedding',
        'norm_scale',
    }
    names, paras, para_groups = filter_params(model, nowd_keys=nowd_keys)

    opt_kw = {
        'lr': 0.0001,
        'betas': (0.9, 0.95),
        'weight_decay': 0,
        'fused': True
    }
    max_psnr = 0

    optimizer = torch.optim.AdamW(params=para_groups, **opt_kw)
    vae_opt = AmpOptimizer(
        fp16=False, optimizer=optimizer, model_maybe_fsdp=model,
        grad_clip=tclip, n_gradient_accumulation=ac, zero=0, bf16=False,
    )

    for epoch in range(epochs):
        if not only_eval:
            train_loader_with_progress = tqdm(train_dataloader,
                                              desc=f'Epoch [{epoch + 1}/{epochs}]',
                                              leave=False)

            for (inpmri,gtpet) in train_loader_with_progress:
                inpmri, gtpet = inpmri.to(device), gtpet.to(device)
                recon, us, lq = model(gtpet)
                Lrec = F.l1_loss(recon, gtpet) + lq
                grad_norm_g, scale_log2_g, loss = vae_opt.backward_clip_step(stepping=True, loss=Lrec)
                train_loader_with_progress.set_postfix(loss=f"{loss.item():.4f}")
            torch.save(model.state_dict(), vae_save_path)
        if epoch % 1 == 0:
            all_psnr = 0
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            model.eval()
            with torch.no_grad():
                for (inpmri, gtpet,fn) in val_loader_with_progress:
                    inpmri, gtpet = inpmri.to(device), gtpet.to(device)
                    inpmri,gtpet = inpmri.squeeze(0),gtpet.squeeze(0)
                    out, _, _ = model(gtpet)
                    save_tensor_to_nii(out, 'VQVAE.nii')
                    save_tensor_to_nii(gtpet, 'VQVAEGT.nii')
                    psnr_scale = calculate_psnr(gtpet, out)
                    all_psnr += psnr_scale
                    val_loader_with_progress.set_postfix(psnr=f"{psnr_scale:.4f}")
                if all_psnr > max_psnr:
                    torch.save(model.state_dict(), vae_save_path)

            print(psnr_scale)


