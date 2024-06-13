import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from net import DehazeDiffusion, Trainer, Unet

from torch import nn
device_ids = [0]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

image_size = 400

model = Unet(
    dim = 64,
    out_dim = 2,
    dim_mults = (1, 2, 4, 8)
).to(device)
diffusion = DehazeDiffusion(model, total_slices = 3).to(device)

trainer = Trainer(
    diffusion,
    '../../dataset/SOTS/SOTS/indoor/hazy',
    '../../dataset/SOTS/SOTS/indoor/hazy', 
    'test',

    image_size,
    batch_size = 1,
    total_step = 160001,
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    save_every = 8000,
    output = 1,
    device_ids=[0]
)
trainer.load(20)
trainer.test()

