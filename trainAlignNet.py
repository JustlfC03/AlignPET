from model.our.model import train_trans,main

#from model.our.gmodel import main

from trainmodel.trainVQVAE import main_train as train_VQVAE
from trainmodel.trainmutitrans import main_train as train_VAR
from trainmodel.trainmutitrans import get_mutie_scale

from trainmodel.train import train_VAE
# from dataloader.dataloader2D import myDataset3
from dataloader.getdata import get_dataloader
#from dataloader.DWT import get_DWT
import os
import sys
import torch

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(ROOT_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)




data_dir = 'datadir'
var_condition_dir = data_dir + '/VARcondition'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs_VAR = 10
epochs_WARPNet = 200
epochs_VQVAE = 50

train_batch_size2D = 32
val_batch_size2D = 1#28
train_batch_size3D = 1
val_batch_size3D = 1

VQVAE_save_path = ''

modality = ''

VAR_save_path = 'VAR'+modality+'.pth'
Fine_generative_save_path = 'Fine_generative_weightnew'+modality+'.pth'
littal_VAE_path = 'littal_VAE_weight'+modality+'.pth'



vqvae_paras_path = None#VQVAE_save_path
pre_VAR_path = VAR_save_path


if __name__ == '__main__':
    #train VAR
    train_dataloader2D,val_dataloader2D = get_dataloader(data_dir,train_batch_size2D,val_batch_size2D,dimension=2,modality=modality)
    train_VQVAE(train_dataloader2D, val_dataloader2D, epochs_VQVAE,device,vae_save_path=VQVAE_save_path,only_eval=False)
    train_VAR(train_dataloader2D, val_dataloader2D, epochs_VAR, device, resume=False, model_save_path=VAR_save_path, vae_local_path=VQVAE_save_path,mriVAE_path=littal_VAE_path,pre_VAR_path=pre_VAR_path)
    train_dataloader2D,val_dataloader2D = get_dataloader(data_dir,val_batch_size2D,val_batch_size2D,dimension=2,train_suffer = False,modality=modality)
    get_mutie_scale(train_dataloader2D, val_dataloader2D, device, model_save_path=VAR_save_path, vae_local_path=VQVAE_save_path, mriVAE_path=littal_VAE_path, muti_scale_path=data_dir,modality=modality)

    
    train_dataloader3D,val_dataloader3D = get_dataloader(data_dir,train_batch_size3D,val_batch_size3D,dimension=3,VAR_condition_path = var_condition_dir,num_workers=4,modality=modality)
    main(train_dataloader3D, val_dataloader3D, epochs_WARPNet, device, resume =False,model_save_path=Fine_generative_save_path,only_eval=False)
