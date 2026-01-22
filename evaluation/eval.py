import numpy as np
import torch
import os
import nibabel as nib

# def calculate_nmse_nmae(gt, recon):
#     """
#     计算真实值(gt)和重建值(recon)之间的NMSE和NMAE

#     参数:
#     gt (numpy.ndarray): 真实值数组
#     recon (numpy.ndarray): 重建值数组

#     返回:
#     tuple: (NMSE, NMAE) 两个指标值
#     """
#     # 确保输入是numpy数组

#     gt = gt.squeeze()[1:-1, ...]
#     recon = recon.squeeze()[1:-1, ...]

#     y = np.asarray(gt, dtype=np.float64)
#     y_hat = np.asarray(recon, dtype=np.float64)

#     # 验证形状匹配
#     if y.shape != y_hat.shape:
#         raise ValueError(f"形状不匹配: gt {y.shape} vs recon {y_hat.shape}")

#     # 展平数组以便计算
#     y_flat = y.ravel()
#     y_hat_flat = y_hat.ravel()

#     # 计算NMSE
#     numerator_nmse = np.sum((y_flat - y_hat_flat) ** 2)
#     denominator_nmse = np.sum(y_flat ** 2)

#     # 避免除以零
#     if denominator_nmse == 0:
#         nmse = 0.0 if numerator_nmse == 0 else np.inf
#     else:
#         nmse = numerator_nmse / denominator_nmse

#     # 计算NMAE
#     numerator_nmae = np.sum(np.abs(y_flat - y_hat_flat))
#     denominator_nmae = np.sum(np.abs(y_flat))

#     # 避免除以零
#     if denominator_nmae == 0:
#         nmae = 0.0 if numerator_nmae == 0 else np.inf
#     else:
#         nmae = numerator_nmae / denominator_nmae

#     return nmse, nmae


from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_nmse_nmae(gt, recon):
    y = np.asarray(gt, dtype=np.float64)
    y_hat = np.asarray(recon, dtype=np.float64)

    if y.shape != y_hat.shape:
        raise ValueError(f"形状不匹配: gt {y.shape} vs recon {y_hat.shape}")

    y_flat = y.ravel()
    y_hat_flat = y_hat.ravel()

    numerator_nmse = np.sum((y_flat - y_hat_flat) ** 2)
    denominator_nmse = np.sum(y_flat ** 2)

    if denominator_nmse == 0:
        nmse = 0.0 if numerator_nmse == 0 else np.inf
    else:
        nmse = numerator_nmse / denominator_nmse

    numerator_nmae = np.sum(np.abs(y_flat - y_hat_flat))
    denominator_nmae = np.sum(np.abs(y_flat))

    if denominator_nmae == 0:
        nmae = 0.0 if numerator_nmae == 0 else np.inf
    else:
        nmae = numerator_nmae / denominator_nmae

    return nmse, nmae


def calculate_psnr_ssim(gt, recon,data_range):
    gt = np.array(gt)
    recon = np.array(recon)
    gt = np.clip(gt, 0, 1)
    recon = np.clip(recon, 0, 1)
    psnr_value = psnr(gt, recon, data_range=data_range)

    # 计算SSIM
    if gt.ndim == 4:  # batch, height, width, channels
        ssim_value = ssim(gt.squeeze(), recon.squeeze(), channel_axis=-1, data_range=data_range)
    elif gt.ndim == 3:  # height, width, channels
        if gt.shape[-1] == 1:  # 单通道
            ssim_value = ssim(gt.squeeze(), recon.squeeze(), data_range=data_range)
        else:  # 多通道
            ssim_value = ssim(gt, recon, channel_axis=-1, data_range=data_range)
    elif gt.ndim == 2:  # height, width (灰度图)
        ssim_value = ssim(gt, recon, data_range=data_range)
    else:
        raise ValueError(f"Unsupported image dimensions: {gt.ndim}")

    return psnr_value, ssim_value




def save_tensor_to_nii(tensor, name,base_dir=None):
    tensor = tensor.squeeze()
    if base_dir is None:
        base_dir = ''
    os.makedirs(base_dir, exist_ok=True)
    data = tensor.cpu()
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    if data.ndim == 3:
        data = data.numpy()
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_dir, name)
        nib.save(img, save_path)
    if data.ndim == 4:
        data = data[:,1,:,:]
        data = data.numpy()
        single_data = data.squeeze()
        single_data = np.transpose(single_data, [1, 2, 0])
        img = nib.Nifti1Image(single_data, affine=np.eye(4))
        save_path = os.path.join(base_dir, name)
        nib.save(img, save_path)


