import os
import sys

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


from .dataloader3D import myDataset3D
from .dataloader2D import myDataset3_slice,myDataset3_wholedata
from torch.utils.data import Dataset, DataLoader

def get_dataloader(data_path,train_batch_size,val_batch_size,num_workers=1,dimension=3,VAR_condition_path = None,Need_VAR = False,train_suffer=True,modality = 'T1_q'):
    if dimension != 2 and dimension != 3:
        raise ValueError('dimension must be 2 or 3')
    if dimension == 3:
        if VAR_condition_path is None and Need_VAR:
            raise Warning("VAR_condition_path cannot be None for 3D data processing. Please provide a valid path.")
        train_dataset = myDataset3D(data_path,var_dir=VAR_condition_path ,mode='train',modality=modality)
        val_dataset = myDataset3D(data_path, var_dir=VAR_condition_path,mode='test',modality=modality)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    if dimension == 2:
        train_dataset = myDataset3_slice(data_path,mode='train_slice',modality=modality)
        if train_suffer == False:
            train_dataset = myDataset3_wholedata(data_path, mode='train',modality=modality)
        val_dataset = myDataset3_wholedata(data_path, mode='test',modality=modality)
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_suffer, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader