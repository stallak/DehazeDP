from ctypes import util
from email.mime import image
import math
from operator import mod
from pyexpat import model

from zmq import device
import torch
import os
from tqdm import tqdm
from torch import nn, einsum
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
from torch.optim import Adam
from einops import rearrange
from inspect import isfunction
from functools import partial
import numpy as np
from tensorboardX import SummaryWriter
import random
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = 1,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        resnet_block_groups = 8,
        learned_variance = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, out_dim, 1),
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)



class DehazeDiffusion(nn.Module):
    def __init__(self, esti_fn, total_slices = 5) -> None:
        super().__init__()
        self.esti_fn = esti_fn
        self.total_slices = total_slices
        self.depth_slices = [i / total_slices for i in range(total_slices+1)]
        self.tanh = nn.Tanh()
    
    def predict_xk_from_x0(self, x_start, k, depth):
        depth = depth
        if k == 0:
            ds = 0.0000001
        else:
            ds = self.depth_slices[k]
        ones = (torch.ones(depth.shape) * (ds)).to(depth.device)
        ones = (torch.ones(depth.shape) * (self.depth_slices[k])).to(depth.device)
        onesm1 = (torch.ones(depth.shape) * (self.depth_slices[k+1])).to(depth.device)
        mask = torch.sign(torch.div(depth, ones, rounding_mode='trunc'))
        maskm1 = torch.sign(torch.div(depth, onesm1, rounding_mode='trunc'))
        beta = (torch.rand(1) * 1.2 + 0.6).to(depth.device)

        t = (- mask * beta * (depth - ones)).exp()
        tm1 = (- beta * maskm1 * (depth - onesm1)).exp()
        A = torch.ones(x_start.shape).to(depth.device)
        A = (A.transpose(1, 3) * (torch.rand(3).to(depth.device) * 0.3 + 0.7)).transpose(1, 3)
        xk = x_start * t + A * (1 - t)
        return t / tm1, xk
    
    def p_losses(self, img, k, depth, depthk, step):
        t, xk = self.predict_xk_from_x0(img, k[0], depth)
        model_out = self.esti_fn(xk, depthk)
        loss = F.mse_loss(model_out, t)
        return loss
    @torch.no_grad()
    def get_hazy(self, img, depth, beta, A):
        t = (-beta * depth).exp()
        return img * t + A * (1 - t)

    def get_hazy_2(self, img, depths, betas, As):
        for i in range(self.total_slices):
            beta = betas[i]
            A = As[i]
            depth = depths[i]
            t = (-beta * depth).exp()
            img = img * t + A * (1 - t)
        return img

    def get_depth_slices(self, depth):
        device = depth.device
        slices = []
        for k in range(self.total_slices):
            ones = (torch.ones(depth.shape)).to(device) / self.total_slices
            onesk = (torch.ones(depth.shape) * (self.depth_slices[k])).to(device)
            onesm1 = (torch.ones(depth.shape) * (self.depth_slices[k+1])).to(device)
            mask = torch.sign(torch.div(depth, onesk, rounding_mode='trunc'))
            maskm1 = torch.sign(torch.div(depth, onesm1, rounding_mode='trunc'))
            mask = mask - maskm1
            slices.append(mask * (depth - onesk) + maskm1 * ones)
        return slices

    @torch.no_grad()
    def get_t(self, depth, beta, k):
        device = depth.device
        ones = (torch.ones(depth.shape)).to(device) / self.total_slices
        onesk = (torch.ones(depth.shape) * (self.depth_slices[k])).to(device)
        onesm1 = (torch.ones(depth.shape) * (self.depth_slices[k+1])).to(device)
        mask = torch.sign(torch.div(depth, onesk, rounding_mode='trunc'))
        maskm1 = torch.sign(torch.div(depth, onesm1, rounding_mode='trunc'))
        mask = mask - maskm1
        t = (-beta * (mask * (depth - onesk) + maskm1 * ones)).exp()
        return t

    def t_losses(self, img, k, depth, depthk, beta):
        model_out = self.esti_fn(img, depthk)
        t_value = self.get_t(depth, beta, k)
        return F.mse_loss(model_out, t_value)

    @torch.no_grad()
    def denoise(self, xk, k):
        b, c, h, w = xk.shape
        k = self.depth_slices[k]
        t = torch.ones(b) * k
        t = t.to(xk.device)
        model_out = self.esti_fn(xk, t)
        t_out = model_out[:, :1, :, :]
        A = self.tanh(model_out[:, 1:, :, :])
        
        return (A - xk) * t_out, A
        
    @torch.no_grad()
    def sample(self, x):
        for i in range(0, self.total_slices):
            x ,A = self.denoise(x, i)
            x = A - x
        return x


    def get_xk_t_mask(self, img, depth, depthkn, beta, k, A):
        b, c, h, w = depth.shape
        device = depth.device
        ones = depthkn.repeat((w, c, h, 1)).transpose(0, 3)
        onesk = ones * (k)
        onesm1 = ones * (k+1)
        if k > 0:
            maskm = torch.sign(torch.div(depth, onesk, rounding_mode='trunc'))
        else:
            maskm = torch.ones((depth.shape)).to(device)
        maskm1 = torch.sign(torch.div(depth, onesm1, rounding_mode='trunc'))
        mask = maskm - maskm1
        t = (-beta * (mask * (depth - onesk) + maskm1 * ones)).exp()
        t_ = (-beta * maskm * (depth - onesk)).exp()
        xk = img * t_ + A * (1 - t_)
        return xk, t

    def forward(self, img, depth, step=0, dataset_type='indoor'):
        b, c, h, w, device= *img.shape, img.device
        loss = 0
        lossa = 0
        losst = 0

        if dataset_type == 'indoor':
            As = [torch.ones(depth.shape).to(device) * (torch.rand(1) * 0.3 + 0.7).to(device) for i in range(self.total_slices)]
            betas = [(torch.rand(1) * 1.2 + 0.6).to(depth.device) for i in range(self.total_slices)]
        elif dataset_type == 'outdoor':
            As = [torch.ones(depth.shape).to(device) * random.choice([0.8, 0.85, 0.9, 0.95, 1]) for i in range(self.total_slices)]
            betas = [random.choice([0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]) for i in range(self.total_slices)]

        depths = self.get_depth_slices(depth)
        depths.reverse()
        hazy = self.get_hazy_2(img, depths, betas, As)

        for i in range(self.total_slices):
            depthk = torch.tensor([k.max() for k in depth]).to(device)
            depthkn = depthk / self.total_slices
            depthk = depthk / self.total_slices * (i + 1)
            model_out = self.esti_fn(hazy, depthk)
            t_out = model_out[:, :1, :, :]
            A_out = self.tanh(model_out[:, 1:, :, :]).repeat((1, 3, 1, 1))
            A = As.pop()
            beta = betas.pop()
            depth = depths.pop()
            t = (-beta * depth).exp()
            hazy = A_out - (A_out - hazy) * t_out
            losst += (F.mse_loss(t_out, 1/t)) / self.total_slices
            lossa += (F.mse_loss(A_out[:, :1, :, :], A)) / self.total_slices
        loss3 = F.mse_loss(hazy, img)
        loss = lossa+losst+loss3
        return loss, lossa.item(), losst.item(), loss3.item()


class Dataset(data.Dataset):
    def __init__(self, folder1, folder2, image_size, exts = ['jpg', 'jpeg', 'png'], type = 'indoor'):
        super().__init__()
        self.folder1 = folder1
        self.folder2 = folder2
        self.image_size = image_size
        self.type = type
        self.paths = [p for ext in exts for p in Path(f'{folder1}').glob(f'**/*.{ext}')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        name = os.path.basename(path).split('.')[0]
        depth = os.path.join(self.folder2, name + '.png')
        depth = Image.open(depth)

        img = self.transform(img)
        depth = self.transform(depth)
        

        return {'image':img, 'depth':depth, 'label':os.path.basename(path)}

class Trainer(object):
    def __init__(self, model, folder1, folder2, dataset_type, image_size, batch_size, total_step, gradient_accumulate_every, train_lr, save_every, device_ids, output) -> None:
        super().__init__()
        self.model = model.cuda()

        self.image_size = image_size
        self.batch_size = batch_size
        self.total_step = total_step
        self.step = 0
        self.gradient_accumulate_every = gradient_accumulate_every
        self.opt = Adam(model.parameters(), lr=train_lr)
        self.save_every = save_every
        self.output = output
        self.dataset_type = dataset_type

        self.ds = Dataset(folder1, folder2, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=8))

    def load(self, milestone):
        return 0
        data = torch.load(f'./results/result1/model-{milestone}.pt')
        self.model.module.load_state_dict(data)
        self.step = milestone * 1000
        print('load success')



    def test(self):
        dl = data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        with tqdm(total=len(dl)) as pbar:
            pbar.set_description('Processing:')
            for i, d in enumerate(dl):
                hazy = d['image'].cuda()
                name = d['label']

                with torch.no_grad():
                    out = self.model.sample(hazy)
                for i in range(out.size(0)):
                    utils.save_image(out[i], 'test/'+name[i])
                    pbar.update(1)

    def train(self):
        writer = SummaryWriter(f'log/log{self.output}')
        with tqdm(initial = self.step, total = self.total_step, ncols = 80) as pbar:
            while self.step < self.total_step:
                for i in range(self.gradient_accumulate_every):
                    self.opt.zero_grad()
                    data = next(self.dl)
                    clear = data['image'].cuda()
                    depth = data['depth'].cuda()
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    loss, lossa, losst, loss3 = self.model(clear, depth, self.step, self.dataset_type)
                    loss.backward()
                    self.opt.step()
                    pbar.set_description(f'loss: {loss.item():.8f}')
                    writer.add_scalar('Train/Loss', loss.item(), self.step)
                    writer.add_scalar('Train/Lossa', lossa, self.step)
                    writer.add_scalar('Train/Losst', losst, self.step)
                    writer.add_scalar('Train/Loss3', loss3, self.step)
                    if self.step % self.save_every == 0 and self.step != 0: 
                        milestone = self.step // self.save_every
                        torch.save(self.model.module.state_dict(), f'./results/result{self.output}/model-{milestone}.pt')
                        print('save milestone', milestone)
                
                self.step += 1
                pbar.update(1)
