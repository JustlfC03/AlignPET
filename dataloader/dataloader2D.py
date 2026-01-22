import os

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm 


def get_all_paths(data_dir, mode,modality='T1_q'):
    mode_path = os.path.join(data_dir, mode)
    mri_paths = []
    pet_paths = []
    filenames = []
    for i in os.listdir(os.path.join(mode_path, modality)):
        mri_path = os.path.join(mode_path, modality, i)
        pet_path = os.path.join(mode_path, 'PET', i)
        mri_paths.append(mri_path)
        pet_paths.append(pet_path)
        filenames.append(i)
    return mri_paths, pet_paths,filenames




class myDataset3_wholedata(Dataset):
    def __init__(self, data_dir, mode,modality):
        self.mri_path, self.pet_path,self.files_name= get_all_paths(data_dir, mode,modality)

    def __len__(self):
        return len(self.pet_path)

    def __getitem__(self, idx):
        mri_path, pet_path = self.mri_path[idx], self.pet_path[idx]
        mri = nib.load(mri_path).get_fdata()
        pet = nib.load(pet_path).get_fdata()

        mri = self._normalize(mri)
        pet = self._normalize(pet)

        mri_tensor = torch.from_numpy(mri).float()
        pet_tensor = torch.from_numpy(pet).float()

        mri_tensor = mri_tensor.permute(2, 0, 1)  # [96, 128, 128]
        pet_tensor = pet_tensor.permute(2, 0, 1)  # [96, 128, 128]

        # ====== 新增：将 [96, 128, 128] 转为 [96, 3, 256, 256] ======
        # 添加通道维 -> [96, 1, 128, 128]
        mri_tensor = mri_tensor.unsqueeze(1)
        pet_tensor = pet_tensor.unsqueeze(1)

        # 通道重复3次 -> [64, 3, 128, 128]
        mri_tensor = mri_tensor.repeat(1, 3, 1, 1)
        pet_tensor = pet_tensor.repeat(1, 3, 1, 1)

        # 零填充到 256x256（中心对齐）
        pad = (64, 64, 64, 64)  # (left, right, top, bottom)
        mri_tensor = torch.nn.functional.pad(mri_tensor, pad)
        pet_tensor = torch.nn.functional.pad(pet_tensor, pad)

        # 现在 mri_tensor 和 pet_tensor 都是 [64, 3, 256, 256]
        return mri_tensor, pet_tensor, self.files_name[idx]

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min() + 1e-8)  # 避免除以0



class myDataset3_slice(Dataset):
    def __init__(self, data_dir, mode, modality):
        self.mri_dir = os.path.join(data_dir, mode,modality)     
        self.pet_dir = os.path.join(data_dir, mode, 'PET')    

        # 获取所有 .npy 文件名（不含扩展名）
        mri_files = [f for f in os.listdir(self.mri_dir) if f.endswith('.npy')]
        pet_files = [f for f in os.listdir(self.pet_dir) if f.endswith('.npy')]

        print(len(mri_files), len(pet_files))

        # 确保文件名一致（如 sub-001_0.npy）
        self.file_names = sorted(set(mri_files) & set(pet_files))
        assert len(self.file_names) > 0, "No matching MRI/PET files found!"

        print(f"✅ Loaded {len(self.file_names)} paired slices from {data_dir}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]

        # Load MRI slice
        mri_path = os.path.join(self.mri_dir, fname)
        mri_array = np.load(mri_path)  # shape (3, 256, 256), dtype float32
        mri_tensor = torch.from_numpy(mri_array).float()  # 转为 float32 Tensor

        

        # Load PET slice
        pet_path = os.path.join(self.pet_dir, fname)
        pet_array = np.load(pet_path)  # shape (3, 256, 256), dtype float32
        pet_tensor = torch.from_numpy(pet_array).float()

        return mri_tensor, pet_tensor