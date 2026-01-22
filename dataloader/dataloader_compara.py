import os
import torch
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as F1
from torch.utils.data import Dataset

def get_all_paths_var(data_dir, var_dir, mode):
    mode_path = os.path.join(data_dir, mode)
    var_path = os.path.join(var_dir, mode)
    mri_paths = []
    pet_paths = []
    var_conditionpath = []
    for i in os.listdir(os.path.join(mode_path, 'image')):
        mri_path = os.path.join(mode_path, 'iamge', i)
        pet_path = os.path.join(mode_path, 'mask', i)
        #pet_path = pet_path.replace('MRI', 'PET')
        mri_paths.append(mri_path)
        pet_paths.append(pet_path)
    for i in os.listdir(var_path):
        var_conditionpath.append(os.path.join(var_path, i))
    return mri_paths, pet_paths, var_conditionpath


def load_mri_PET_MC_3D(mri_paths, pet_paths, varconditionpath):
    mris = []
    pets = []
    varconditions = []
    for i in range(len(mri_paths)):
        mri_path, pet_path = mri_paths[i], pet_paths[i]
        mri = nib.load(mri_path).get_fdata()
        pet = nib.load(pet_path).get_fdata()
        var_condition = torch.load(varconditionpath[i])

        mris.append(mri)
        pets.append(pet)
        varconditions.append(var_condition)

    return mris, pets, varconditions


class myDataset3D(Dataset):
    def __init__(self, data_dir, var_dir, mode):
        mri_path, pet_path, var_conditionpath = get_all_paths_var(data_dir, var_dir, mode)
        self.mris, self.pets, self.varconditions = load_mri_PET_MC_3D(mri_path, pet_path, var_conditionpath)

    def __len__(self):
        return len(self.mris)

    def __getitem__(self, idx):
        # 获取原始数据
        mri = self.mris[idx].astype(np.float32)  # 转换为float32类型
        pet = self.pets[idx].astype(np.float32)

        def normalize(data):
            data_min = np.min(data)
            data_max = np.max(data)
            # 避免除零错误
            denominator = data_max - data_min + 1e-8
            return (data - data_min) / denominator

        # 应用归一化
        mri = mri.transpose((2, 0, 1))
        pet = pet.transpose((2, 0, 1))

        mri = F1.resize(torch.from_numpy(mri), size=(256, 256))
        pet = F1.resize(torch.from_numpy(pet), size=(256, 256))

        mri = normalize(mri.numpy())
        pet = normalize(pet.numpy())

        mri_tensor = torch.from_numpy(mri).unsqueeze(0)
        pet_tensor = torch.from_numpy(pet).unsqueeze(0)
        return mri_tensor, pet_tensor