import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from net import Unet, DehazeDiffusion, Trainer

device_ids = [0]
image_size = 32

dataset = 'indoor'
exp_index = 1
if dataset == 'indoor':
    clear_path = '../../dataset/RESIDE/ITS/clear/'
    depth_path = '../../dataset/NYUdepth/depth/'
elif dataset == 'outdoor':
    clear_path = '../dataset/RESIDE/OTS/clear/'
    depth_path = '../dataset/RESIDE/OTS/depth_images/'
elif dataset == 'test':
    clear_path = '../dataset/SOTS/SOTS/indoor/gt'
    depth_path = '../dataset/NYUdepth/depth/'

model = Unet(
    dim = 64,
    out_dim = 2,
    dim_mults = (1, 2, 4, 8)
)
diffusion = DehazeDiffusion(model, total_slices = 3)

trainer = Trainer(
    diffusion,
    clear_path,
    depth_path,
    dataset,
    image_size,
    batch_size = 4,
    total_step = 100,
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    save_every = 10000,
    device_ids = device_ids,
    output = exp_index
)

trainer.train()
