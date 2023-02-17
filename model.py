import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Sine(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_res=False, omega_0=30):
        super(Sine, self).__init__()
        # omega for sine activation
        self.omega_0 = omega_0
        # is first layer
        self.is_first = is_first
        # is res block
        self.is_res = is_res

        # linear layer
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias)
        self.init_weights(self.linear)
    
    def init_weights(self, layer):
        with torch.no_grad():
            if self.is_first:
                layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                    np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, input):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))



class CondSIREN(nn.Module):
    def __init__(self, n_emb, norm_p = None, inter_fn = None, first_omega_0=30, hidden_omega_0=30, D=8, z_dim = 64, in_feat=2, out_feat=3, W=256, with_res=True, with_norm=True):
        # n_emb is the number of images in the dataset
        # z_dim is the length of z, the code (in vector) for each image, randomly initialized
        
        super().__init__()
        # regularize z to the same scale later on for interpolation tasks
        self.norm_p = norm_p
        # nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size
        # for each image, we have a code vector z, which is randomly initialized
        # the length of z is z_dim
        # for example, for input array of shape (x, y), we have output array of shape (x, y, z_dim)
        if self.norm_p is None or self.norm_p == -1:
            self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=z_dim)
        else:
            self.emb = nn.Embedding(num_embeddings=n_emb, embedding_dim=z_dim, max_norm=1.0, norm_type=norm_p)
        
        # the sine layers
        # for i in range(D+1):
        for i in range(D):
            if i == 0:
                # in_feat = (x, y), concatenated with image code vector z
                # output to W, default length 256 for hidden layers
                layer = Sine(in_feat + z_dim, W, is_first=True, is_res=False, omega_0=first_omega_0)
            else:
                # Sine(in=256, out=256) for all hidden layers
                layer = Sine(W, W, is_first=False, is_res=with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(W))
            setattr(self, f"layer_{i+1}", layer)
        
        # out dim = 3 (rgb)
        final_linear = nn.Linear(W, out_feat, bias=True)
        # TODO: why this initialization?
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / W) / hidden_omega_0,  np.sqrt(6 / W) / hidden_omega_0)
            # Use of identity?
        self.final_rgb = nn.Sequential(final_linear, nn.Identity())
        self.D = D
        self.inter_fn = inter_fn
    
    def normalize_z(self, z):
        if self.norm_p == -1:
            z = z / torch.max(z, dim=-1)[0]
        elif self.norm_p is not None:
            z = F.normalize(z, p=self.norm_p, dim=-1)
        else:
            z = z
        return z

    def forward_with_z(self, x, z):
        xyz_ = torch.cat([x, z.unsqueeze(1).repeat(1, x.shape[1], 1)], dim = -1)
        for i in range(self.D):
            xyz_ = getattr(self, f'layer_{i+1}')(xyz_)
        rgb = self.final_rgb(xyz_)
        return rgb
    
    def ret_z(self, ind):
        z = self.emb(ind)
        z = self.normalize_z(z)
        return z
    
    def get_all_Z_mat(self):
        Z_mat = self.emb.weight
        return self.normalize_z(Z_mat)
    
    def forward(self, x, ind, ret_z=False):
        z = self.emb(ind)
        z = self.normalize_z(z)
        rgb = self.forward_with_z(x, z)
        if ret_z:
            return rgb, z
        else:
            return rgb


class VIINTER(CondSIREN):
    def mix_forward(self, xy_grid_flattened, batch_size=4, chunked=False):
        N = self.emb.num_embeddings
        all_inds = torch.arange(0, N).type(torch.LongTensor).to(xy_grid_flattened.device)
        # embed and normalize z vector
        zs = self.emb(all_inds)
        zs = self.normalize_z(zs)

        rand_inds = torch.randint(0, N, size=(batch_size * 2, 1)).long().squeeze()

        # ??
        slt_zs = zs[rand_inds].reshape(batch_size, 2, -1)
        # rand_like returns a tensor with the same size as input that is filled with random numbers from a uniform distribution on the interval [0, 1)
        alphas = torch.rand_like(slt_zs[:, 0:1, 0:1])
        z = self.inter_fn(a=slt_zs[:, 0], b=slt_zs[:, 1], t=alphas).squeeze(1)
        x = xy_grid_flattened.repeat(batch_size, 1, 1)

        # ??
        if chunked:
            rgb = torch.zeros((x.shape[0], x.shape[1], 3), device=x.deivce)
            _p = 8192 * 1
            for ib in range(0, rgb.shape[1], _p):
                for ip in range(0, rgb.shape[1], _p):
                    rgb[ib:ib+1, ip:ip+_p] = self.forward_with_z(x[ib:ib+1, ip:ip+_p], z)
        else:
            rgb = self.forward_with_z(x, z)
        
        rand_inds = rand_inds.reshape(batch_size, 2)
        return rgb, rand_inds[:, 0], rand_inds[:, 1], alphas, z