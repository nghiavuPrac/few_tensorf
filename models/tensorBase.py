import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret 
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret

def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None, type='submission', device='cpu'):
  if max_visible is None:
    # default FreeNeRF
    dv = 5
    if current_iter < total_reg_iter:
      freq_mask = torch.zeros(pos_enc_length).to(device)  # all invisible
      ptr = pos_enc_length / dv * current_iter / total_reg_iter + 1 
      ptr = ptr if ptr < pos_enc_length / dv else pos_enc_length / dv
      int_ptr = int(ptr)
      freq_mask[: int_ptr * dv] = 1.0  # assign the integer part
      freq_mask[int_ptr * dv : int_ptr * dv + 3] = (ptr - int_ptr)  # assign the fractional part
      return torch.clamp(freq_mask, 1e-8, 1 - 1e-8)
    else:
      return torch.ones(pos_enc_length).to(device)
  else:
    # For the ablation study that controls the maximum visible range of frequency spectrum
    freq_mask = torch.zeros(pos_enc_length).to(device)
    freq_mask[: int(pos_enc_length * max_visible)] = 1.0
    return freq_mask

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128, encoder=None):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC  = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe   = viewpe
        self.feape    = feape

        self.encoder  = encoder
        self.fea_encoder = self.encoder(0, feape)
        self.view_encoder = self.encoder(0, viewpe)

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, step, total_step):
        indata = [features, viewdirs]
        if self.feape > 0:
            encode = self.fea_encoder(features)

            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]
            
        if self.viewpe > 0:
            encode = self.view_encoder(viewdirs)

            if step == -1:
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128, encoder=None):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe

        self.encoder  = encoder
        self.pos_encoder = self.encoder(0, pospe)
        self.view_encoder = self.encoder(0, viewpe)

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, step, total_step):
        indata = [features, viewdirs]
        if self.pospe > 0:
          
            """if len(pts) == 2:
                mean = pts[0]
                var = pts[1]
                encode, _ = self.pos_encoder(mean, var)
            else:
                pts = pts[0]
                encode = self.pos_encoder(pts)"""
            encode = self.pos_encoder(pts)

            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        if self.viewpe > 0:

            encode = positional_encoding(viewdirs, self.viewpe)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, pospe=6, viewpe=6, feape=6, featureC=128, encoder=None):
        super(MLPRender, self).__init__()

        self.in_mlpC = (2*pospe*3) + (3+2*viewpe*3) + (2*feape*inChanel) + inChanel #
        self.pospe = pospe
        self.viewpe = viewpe
        self.feape = feape
        

        self.encoder = encoder
        self.pos_encoder = self.encoder(0, pospe)
        self.fea_encoder = self.encoder(0, feape)
        self.view_encoder = self.encoder(0, viewpe)

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, step, total_step):
        indata = [features, viewdirs]

        if self.pospe > 0:
            encode = self.pos_encoder(pts)

            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        if self.feape > 0:
            encode = positional_encoding(features, self.feape)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

"""class NeRF(nn.Module):
    def __init__(self,
                 D,
                 W,
                 d_in,
                 d_in_view,
                 pospe,
                 viewpe,
                 feape,
                 inChanel,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        self.in_mlpC = (2*pospe*3) + (3+2*viewpe*3) + (2*feape*inChanel) + inChanel #
        self.pospe = pospe
        self.viewpe = viewpe
        self.feape = feape
        
        self.encoder = encoder
      
        if pospe > 0:
            self.pos_encoder = self.encoder(0, pospe)

        if viewpe > 0:
            self.view_encoder = self.encoder(0, viewpe)

        if feape > 0:
            self.fea_encoder = self.encoder(0, feape)

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_mlpC, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):

        indata = [features, viewdirs]

        if self.pospe > 0:
            encode = self.pos_encoder(pts)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]
        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]
        if self.feape > 0:
            encode = positional_encoding(features, self.feape)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(encode.shape[1], step, total_step, None, device=encode.device).tile((encode.shape[0], 1))
                indata += [encode*mask]
        mlp_in = torch.cat(indata, dim=-1)


        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False"""

class TensorBase(torch.nn.Module):
    def __init__(self, args, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.args = args

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.encoder = PositionalEncoding
        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device, self.encoder)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device, encoder):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC, encoder).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC, encoder).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, pos_pe, view_pe, fea_pe, featureC, encoder).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'device': self.device
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox
    
    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples
                
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def forward(self, rays_chunk, step=-1, total_step=0, white_bg=True, is_train=False, ndc_ray=False, mip=False, randomized=True, ray_shape='cone', N_samples=-1):

        # sample points
        origins = rays_chunk[:, :3]
        viewdirs = rays_chunk[:, 3:6]

        ret = []      
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        
        exp_viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], -1, total_step)

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask], step, total_step)

            valid_rgbs = self.renderModule(xyz_sampled[app_mask], exp_viewdirs[app_mask], app_features, step, total_step)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        ret.append((rgb_map, rgb, depth_map, weight))

        return rgb_map, rgb, depth_map, alpha #, rgb, alpha #, sigma, weight, bg_weight