import time
from typing import List, Optional, Tuple, Union

import torch
import os
import sys
from trainmodel.train import train_VAE

from trainmodel.trainVQVAE import main_train as train_VQVAE
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

from model.VAR.models import VAR, VQVAE, VectorQuantizer2
from model.VAR.models.varnew import VARrecon
from model.VAR.models.VAEtry import VAE
from pprint import pformat

# from dataloader.dataloader2D import myDataset3

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
            self,
            mixed_precision: int,
            optimizer: torch.optim.Optimizer, names: List[str], paras: List[torch.nn.Parameter],
            grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1

        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True,
                                          dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16,
                                          cache_enabled=True)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11,
                                                    growth_interval=1000) if self.using_fp16_rather_bf16 else None  # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None

        self.optimizer, self.names, self.paras = optimizer, names, paras  # paras have been filtered so everyone requires grad
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
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.:  # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    self.scaler.update(new_scale=32768.)
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
                orig_norm = self.optimizer.global_grad_norm

            self.optimizer.zero_grad(set_to_none=True)

        return loss

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


# def save_tensor_to_nii(tensor):
#     save_path = '/home/cyf/CAPL/VAR-mainbase/VARrecon/mid3/mid3.nii'
#     data = tensor.squeeze().cpu().numpy()
#     data = np.transpose(data, [1, 2, 0])
#     img = nib.Nifti1Image(data, affine=np.eye(4))  # identity affine 矩阵，可替换为真实 affine
#     nib.save(img, save_path)
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


class VARTrainer(object):
    def __init__(
            self, device, vae_local: VQVAE, var_wo_ddp: VARrecon,
            var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.vae_local, self.quantize_local = vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VARrecon = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums = patch_nums
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    def train_step(
            self, stepping: bool, inp_B3HW: FTen, MRI_latent: Union[ITen, FTen], gtpet, sos_mode
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        B, V = MRI_latent.shape[0], self.vae_local.vocab_size
        self.var_wo_ddp.require_backward_grad_sync = stepping

        with torch.no_grad():
            #print('gtpet_shape',gtpet.shape)
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(gtpet)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            _, logits_BLV, ababab_ = self.var_wo_ddp(x_BLCv_wo_first_l, MRI_latent)

            #print('logits_BLV_shape',logits_BLV.shape,gt_BL.shape)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

        # backward
        loss = self.var_opt.backward_clip_step(loss=loss, stepping=True)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return loss


def compile_model(m, fast):
    if fast == 0:
        return m
    return torch.compile(m, mode={
        1: 'reduce-overhead',
        2: 'max-autotune',
        3: 'default',
    }[fast]) if hasattr(torch, 'compile') else m


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


def main_train(train_dataloader, val_dataloader, epochs, device, resume, model_save_path=None, vae_local_path=None, mriVAE_path=None, pre_VAR_path=None):
    
    train_VAE(train_dataloader, val_dataloader, epochs=2, device=device,model_save_path=mriVAE_path)

    #return None

    if pre_VAR_path is None:
        print('No pre-trained weights from VAR were used.')

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4

    num_classes = 1000
    depth = 16
    shared_aln = False
    attn_l2_norm = True
    flash_if_available = True
    fused_if_available = True,
    fp16: int = 0  # 1: using fp16, 2: bf16
    tlr: float = 0.0001  # lr = base lr * (bs / 256)
    tclip: float = 2.  # <=0 for not using grad clip

    max_psnr_ssim = 0.0

    ac = 1
    sos_mode = 'mri'

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VARrecon(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)

    vae_local: VQVAE = compile_model(vae_local, 0)

    MRI_VAE = VAE().to(device)
    MRI_VAE.load_state_dict(torch.load(mriVAE_path))
    MRI_VAE.eval()

    vae_state_dict = torch.load(vae_local_path, map_location=device)
    

    vae_local.load_state_dict(vae_state_dict, strict=True)
    vae_local.to(device)
    vae_local.eval()

    if pre_VAR_path is not None:
        var_wo_ddp_state_dict = torch.load(pre_VAR_path, map_location=device)
        var_wo_ddp.load_state_dict(var_wo_ddp_state_dict, strict=False)
        var_wo_ddp.to(device)

    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_kw = {
        'lr': tlr,
        'betas': (0.9, 0.95),
        'weight_decay': 0,
        'fused': True
    }
    optimizer = torch.optim.AdamW(params=para_groups, **opt_kw)
    var_optim = AmpOptimizer(
        mixed_precision=fp16, optimizer=optimizer, names=names, paras=paras,
        grad_clip=tclip, n_gradient_accumulation=ac
    )

    trainer = VARTrainer(device,
                         vae_local=vae_local, var_wo_ddp=var_wo_ddp,
                         var_opt=var_optim, label_smooth=0.0,
                         )

    for epoch in range(epochs):
        # ---------- Training phase ----------
        train_loader_with_progress = tqdm(train_dataloader,
                                          desc=f'Epoch [{epoch + 1}/{epochs}]',
                                          leave=False)

        total_train_loss = 0.0
        num_train_batches = 0

        var_wo_ddp.train()
        for (inpmri, gtpet) in train_loader_with_progress:
            inpmri = inpmri.to(device)
            gtpet = gtpet.to(device)

            with torch.no_grad():
                MRI_latent = MRI_VAE.encoder(inpmri)

            batch_loss = trainer.train_step(True, inp_B3HW=inpmri, MRI_latent=MRI_latent, gtpet=gtpet,
                                            sos_mode=sos_mode)
            total_train_loss += batch_loss.item()
            num_train_batches += 1

            train_loader_with_progress.set_postfix(loss=f"{batch_loss.item():.4f}")

        avg_train_loss = total_train_loss / num_train_batches
        print(avg_train_loss)

        if epoch % 1 == 0:
            val_loader_with_progress = tqdm(val_dataloader,
                                            desc=f'Epoch [{epoch + 1}/{epochs}]',
                                            leave=False)
            var_wo_ddp.eval()

            total_psnr = 0.0
            total_nmse = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for (inpmri, gtpet,fn) in val_loader_with_progress:
                    inpmri, gtpet = inpmri.to(device), gtpet.to(device)
                    inpmri,gtpet = inpmri.squeeze(0),gtpet.squeeze(0)
                    inpmri = MRI_VAE.encoder(inpmri)
                    out, MCimages = var_wo_ddp.autoregressive_infer_cfg(inpMRI=inpmri)
                    psnr_scale = calculate_psnr(gtpet, out)
                    total_psnr += psnr_scale
                    num_val_batches += 1
                    save_tensor_to_nii(out, 'mutiscaleOUT.nii')
                    save_tensor_to_nii(gtpet, 'mutiscaleGT.nii')
                    val_loader_with_progress.set_postfix(
                        psnr=f"{psnr_scale:.4f}"
                    )
            avg_psnr = total_psnr / num_val_batches

            if max_psnr_ssim < avg_psnr :
                torch.save(var_wo_ddp.state_dict(), model_save_path)

            print(f"Epoch {epoch + 1} - Val: PSNR={avg_psnr:.4f}")


def get_mutie_scale(train_dataloader, val_dataloader, device, model_save_path=None, vae_local_path=None,
                    mriVAE_path=None, muti_scale_path=None,modality=None):
    
    # vae_local_path = '/data/birth/cyf/output/CAPL/outputzgy/ADNI/weight/VQVAEPET.pth'
    # mriVAE_path = '/data/birth/cyf/output/CAPL/VAR/VAExuanwu.pth'
    # model_save_path = '/data/birth/cyf/output/CAPL/outputzgy/xuanwu/weight/VAR.pth'

    muti_scale_path_train = os.path.join(muti_scale_path, 'VARcondition','train',modality)
    muti_scale_path_val = os.path.join(muti_scale_path, 'VARcondition','test',modality)
    
    if not os.path.exists(muti_scale_path_train):
        os.makedirs(muti_scale_path_train)
    if not os.path.exists(muti_scale_path_val):
        os.makedirs(muti_scale_path_val)
    

    epochs = 1
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    V = 4096
    Cvae = 32
    ch = 160
    share_quant_resi = 4
    num_classes = 1000
    depth = 16
    shared_aln = False
    attn_l2_norm = True
    flash_if_available = True
    fused_if_available = True,
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums).to(device)
    var_wo_ddp = VARrecon(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    ).to(device)

    vae_local: VQVAE = compile_model(vae_local, 0)
    vae_state_dict = torch.load(vae_local_path, map_location=device)
    vae_local.load_state_dict(vae_state_dict, strict=True)
    vae_local.to(device)
    vae_local.eval()

    state_VAE = VAE().to(device)
    state_dict = torch.load(mriVAE_path, map_location='cpu')
    state_VAE.load_state_dict(state_dict)
    state_VAE.eval()

    var_wo_ddp_state_dict = torch.load(model_save_path, map_location=device)
    var_wo_ddp.load_state_dict(var_wo_ddp_state_dict, strict=False)
    var_wo_ddp.to(device)

    val_loader_with_progress = tqdm(val_dataloader,
                                    desc=f'Epoch [{1}/{epochs}]',
                                    leave=False)
    train_loader_with_progress = tqdm(train_dataloader,
                                      desc=f'Epoch [{1}/{epochs}]',
                                      leave=False)

    var_wo_ddp.eval()
    state_VAE.eval()
    save = []
    with torch.no_grad():
        num_val_batches = 0
        for (inpmri, gtpet,files_name) in val_loader_with_progress:
            inpmri = inpmri.squeeze(0)
            gtpet = gtpet.squeeze(0)
            name = files_name[0]
            print(name)
            name = name.split('.')[0]
            name = name + '.npy'
            inpmri, gtpet = inpmri.to(device), gtpet.to(device)
            inpmri,gtpet = inpmri.squeeze(0),gtpet.squeeze(0)
            inpmri = state_VAE.encoder(inpmri)
            out, MCimages = var_wo_ddp.autoregressive_infer_cfg(inpMRI=inpmri)
            #save.append(MCimages)
            MCsave_path = os.path.join(muti_scale_path_val, name)
            print(MCsave_path)
            np.save(MCsave_path, MCimages.detach().cpu().numpy())
            #torch.save(MCimages, MCsave_path)
            print(MCimages.shape)

            psnr_scale = calculate_psnr(gtpet, out)
            num_val_batches += 1
            # save_tensor_to_nii(out, 'outvar.nii')
            # save_tensor_to_nii(gtpet, 'gtvar.nii')
            val_loader_with_progress.set_postfix(
                psnr=f"{psnr_scale:.4f}"
            )
        num_train_batches = 0
        for (inpmri, gtpet,files_name) in train_loader_with_progress:
            inpmri = inpmri.squeeze(0)
            gtpet = gtpet.squeeze(0)
            name = files_name[0]
            print(name)
            name = name.split('.')[0]
            name = name + '.npy'
            inpmri, gtpet = inpmri.to(device), gtpet.to(device)
            inpmri,gtpet = inpmri.squeeze(0),gtpet.squeeze(0)
            inpmri = state_VAE.encoder(inpmri)
            out, MCimages = var_wo_ddp.autoregressive_infer_cfg(inpMRI=inpmri)
            #save.append(MCimages)
            MCsave_path = os.path.join(muti_scale_path_train, name)
            print(MCsave_path)
            np.save(MCsave_path, MCimages.detach().cpu().numpy())
            print(MCimages.shape)

            psnr_scale = calculate_psnr(gtpet, out)
            num_train_batches += 1
            train_loader_with_progress.set_postfix(
                psnr=f"{psnr_scale:.4f}"
            )

