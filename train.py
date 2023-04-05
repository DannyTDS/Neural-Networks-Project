import numpy as np
import os
import clip
import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

from utils import *

class Solver():
    def __init__(self, args, dataset, model, device):
        # log path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        self.log_file = os.join(args.log_dir, args.name+'.txt')

        self.device = device
        self.silent = args.silent
        self.clip = args.clip
        self.clip_im_size = 224
        self.percep_freq = args.percep_freq
        self.save_freq = args.save_freq
        self.save_dir = args.save_dir

        # hyper params
        self.bsize = args.bsize
        self.num_steps = args.iters
        self.save_freq = 1000

        # network components
        self.dataset = dataset
        self.train_loader = infiniteloop(DataLoader(self.dataset, batch_size=self.bsize, shuffle=True, drop_last=False, num_workers=4, pin_memory=True))
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.start_lr, betas=(0.9,0.999), weight_decay=1e-6)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_steps, eta_min=args.final_lr)

        self.min_side = min(self.dataset.hw[0], self.dataset.hw[1])
        self.ds_ratio = self.min_side // 256
        if self.clip != 0.0:
            print("Initializing CLIP model...")
            self.init_clip()
    

    def init_clip(self):
        self.clip_model, self.preprocess = clip.load("ViT-B/32")
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.clip_im_size, self.clip_im_size)),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # precompute clip features
        self.im_feats = []
        self.imgs = []
        with torch.no_grad():
            for im in self.dataset.imgs:
                if self.min_side > 256:
                    im = torchvision.transforms.Resize(self.in_side//self.ds_ratio)(im)
                im = im.to(self.device).unsqueeze(0)
                patches = F.unfold(im, kernel_size = (self.clip_im_size, self.clip_im_size), stride=self.clip_im_size)
                patches = patches.reshape(3, self.clip_im_size, self.clip_im_size, -1).permute(3, 0, 1, 2)
                self.im_feats.append(self.clip_model.encode_image(self.clip_transform(patches)).float().detach().squeeze())
                self.imgs.append(im.to(self.device))
        self.im_feats = torch.stack(self.im_feats, dim=0)
        self.imgs = torch.stack(self.imgs, dim=0)


    def train(self):
        print(f"Start training for {self.num_steps} steps")
        print(f"Batch size: {self.bsize}")
        print(f"Image size: {self.dataset.hw}")

        coords_h = np.linspace(-1, 1, self.dataset.hw[0], endpoint=False)
        coords_w = np.linspace(-1, 1, self.dataset.hw[1], endpoint=False)
        # np.meshgrid returns a list of 2D arrays (x, y), so we stack them to get a 3D array (x, y, 2)
        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), axis=-1)
        # view() flattens the array, contiguous() makes sure the array is stored in a contiguous block of memory
        # unsqueeze(0) adds a dimension at the beginning, so that the array has shape (1, h*w, 2)
        grid_inp = torch.FloatTensor(xy_grid).view(-1, 2).contiguous().unsqueeze(0).to(self.device)

        # some resizing for using clip?
        if self.min_side > 256:
            coords_h_ds = np.linspace(-1, 1, self.dataset.hw[0]//self.ds_ratio, endpoint=False)
            coords_w_ds = np.linspace(-1, 1, self.dataset.hw[1]//self.ds_ratio, endpoint=False)
            xy_grid_ds = np.stack(np.meshgrid(coords_w_ds, coords_h_ds), axis=-1)
            grid_inp_ds = torch.FloatTensor(xy_grid_ds).view(-1, 2).contiguous().unsqueeze(0).to(self.device)

            clip_grid_inp = grid_inp_ds
            clip_hw = [self.dataset.hw[0]//self.ds_ratio, self.dataset.hw[1]//self.ds_ratio]
        
        else:
            clip_grid_inp = grid_inp
            clip_hw = self.dataset.hw

        steps = 0
        # tqdm.trange() is a wrapper around range() that displays a progress bar
        # loop = tqdm.trange(self.num_steps, disable=self.silent)
        for i in range(self.num_steps):
            img, ind = next(self.train_loader)
            # for img,ind in self.train_loader:
            
            # <--- BP with Linter -->
            self.optimizer.zero_grad()
            if self.clip != 0.0:
                if i % self.percep_freq == 0:
                    mix_out, ia, ib, alpha, z = self.model.mix_forward(clip_grid_inp, batch_size=self.bsize)
                    # mix_out, ia, ib, alpha, z = self.model.mix_forward(clip_grid_inp)
                    mix_out = mix_out.view(-1, clip_hw[0], clip_hw[1], 3)

                    patches = F.unfold(mix_out.permute(0, 3, 1, 2), kernel_size=(self.clip_im_size, self.clip_im_size), stride=self.clip_im_size)
                    patches = patches.reshape(3, self.clip_im_size, self.clip_im_size, -1).permute(3, 0, 1, 2)
                    # print(patches)
                    out_emb = self.clip_model.encode_image(self.clip_transform(patches)).float().squeeze()
                    # print(out_emb)
                    mix_emb = (self.im_feats[ia] * (1-alpha)) + (self.im_feats[ib] * alpha)
                    feats_loss = F.mse_loss(out_emb, mix_emb.squeeze()) * self.clip
                    feats_loss.backward()
            self.optimizer.step()

            # img.shape = (batch_size, 3, h, w)
            num_pixels = img.shape[-2] * img.shape[-1]
            # randperm() returns a tensor of random integers from 0 to num_pixels-1
            sind = torch.randperm(num_pixels)[:self.bsize].squeeze()
            # permute() changes the order of the dimensions, reshape() changes the shape of the tensor
            img, ind = img[0].permute(1, 2, 0).reshape(-1, 3)[sind].to(self.device), torch.LongTensor([ind[0]]).to(self.device)

            # <--- BP with Lrecon --->
            self.optimizer.zero_grad()
            out = self.model(grid_inp[:, sind], ind).squeeze()
            mse_loss  = F.mse_loss(out, img)
            mse_loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            psnr = 10 * np.log10(1 / mse_loss.item())
            print("Iter {}:\tPSNR = {:.3f};\tloss = {:.3f}".format(steps, psnr, mse_loss.item()))
            # loop.set_postfix(PSNR = psnr)
            # loop.set_postfix(loss = mse_loss.item())
            steps += 1

            if steps % self.save_freq == 0:
                if self.clip > 0.0:
                    generated = torch.clamp(mix_out[0].detach().cpu(), 0, 1).numpy()
                    Image.fromarray(np.uint8(255 * generated)).save(f"{self.save_dir}/generated_{steps}_mix.png")
                torch.save(self.model.state_dict(), f"{self.save_dir}/model_{steps}.pt")
                # inference part?
                self.model.eval()
                with torch.no_grad():
                    out = torch.zeros((grid_inp.shape[-2], 3))
                    _b = 8192 * 4
                    for ib in range(0, len(out), _b):
                        out[ib:ib+_b] = self.model(grid_inp[:, ib:ib+_b], torch.LongTensor([0]).to(self.device)).cpu()
                self.model.train()
                # clamp() makes sure that all values are between 0 and 1
                generated = torch.clamp(out.view(*self.dataset.hw, 3), 0, 1).numpy()
                Image.fromarray(np.uint8(255 * generated)).save(f"{self.save_dir}/generated_{steps}.png")

                # TODO: left out: GIF generation

        torch.save(self.model.state_dict(), f"{self.save_dir}/model.pt")

        # TODO: left out: unpack GIF frames and save as PNGs

        training_psnr, training_ssim = 0, 0
        for i in range(len(self.dataset)):
            with torch.no_grad():
                out = torch.zeros((grid_inp.shape[-2], 3))
                _b = 8192 * 8
                for ib in range(0, len(out), _b):
                    out[ib:ib+_b] = self.model(grid_inp[:, ib:ib+_b].to(self.device), torch.LongTensor([i]).to(self.device)).cpu()
            generated = torch.clamp(out.view(*self.dataset.hw, 3), 0, 1)
            out = np.uint8(255 * np.clip(generated.numpy(), 0, 1))
            training_mse = F.mse_loss(self.dataset.imgs[i].permute(1, 2, 0), generated).item()
            training_psnr += 10 * np.log10(1 / training_mse)
            training_ssim += ssim(np.clip(generated.numpy(), 0, 1), self.dataset.imgs[i].permute(1, 2, 0).numpy(), channel_axis=2, multichannel=True)
        training_psnr, training_ssim = training_psnr / len(self.dataset), training_ssim / len(self.dataset)
        print(f"Training PSNR: {training_psnr:.2f}, SSIM: {training_ssim:.4f}")