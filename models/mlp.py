import torch
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
    
        freq_bands = (2**torch.arange(freqs).float()).to(
            positions.device
        )  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], )
        )  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


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
            encode = self.pos_encoder(pts)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(
                  encode.shape[1], 
                  step, 
                  total_step, 
                  None, 
                  device=encode.device
                ).tile(
                  (encode.shape[0], 1)
                )
                indata += [encode*mask]

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)
            if step == -1:    
                indata += [encode]
            else:
                mask = get_freq_reg_mask(
                  encode.shape[1], 
                  step, 
                  total_step, 
                  None, 
                  device=encode.device
                ).tile(
                  (encode.shape[0], 1)
                )
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

        self.mlp = torch.nn.Sequential(
          layer1, 
          torch.nn.ReLU(inplace=True), 
          layer2, 
          torch.nn.ReLU(inplace=True), 
          layer3
        )

        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, mask):
        indata = [features, viewdirs]

        if self.pospe > 0:
            encode = self.pos_encoder(pts)

            if mask == None:    
                indata += [encode]
            else:
                indata += [encode*mask['pos']]

        if self.viewpe > 0:
            encode = positional_encoding(viewdirs, self.viewpe)
            if mask == None:
                indata += [encode]
            else:
                indata += [encode*mask['view']]

        if self.feape > 0:
            encode = positional_encoding(features, self.feape)
            if mask == None:
                indata += [encode]
            else:
                indata += [encode*mask['fea']]

        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb