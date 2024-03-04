# 添加系统目录到系统环境变量
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    # with torch.cuda.amp.autocast(enabled=True):
    out = torch.gather(v, index=t, dim=0).to(device).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a.float()

class GaussDiffuser(nn.Module):
    def __init__(self, model, beta_1, beta_T, time_steps):
        super().__init__()
        self.model = model
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.time_steps = time_steps

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, time_steps).double())
        alpha = 1. - self.betas
        alphas_bar = torch.cumprod(alpha, 0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:time_steps]
        # train parameters
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        # sample parameters
        self.register_buffer('coeff1', torch.sqrt(1. / alpha))
        self.register_buffer('coeff2', self.coeff1 * (1. - alpha) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var
    
    @torch.no_grad
    def ddim_sample(self, x_T, ddim_timesteps=50,ddim_eta=0.0,*args):
        batch_size = x_T.shape[0]
        # ddim
        c = self.time_steps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.time_steps, c)))
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_next_seq = [-1] + list(ddim_timestep_seq[:-1])

        device = x_T.device
        # start from pure noise (for each example in the batch)
        sample_img = x_T
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), ddim_timestep_next_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = compute_alpha(self.betas, t)
            alpha_cumprod_t_next = compute_alpha(self.betas, next_t)
            et = self.model(sample_img, t)
            x0_t = (sample_img - et * (1-alpha_cumprod_t).sqrt())/alpha_cumprod_t.sqrt()
            c1 = ddim_eta * ((1-alpha_cumprod_t/alpha_cumprod_t_next) * (1-alpha_cumprod_t_next)/(1-alpha_cumprod_t)).sqrt()
            c2 = ((1-alpha_cumprod_t_next) - c1**2).sqrt()
            sample_img = alpha_cumprod_t_next.sqrt() * x0_t + c1 * torch.randn_like(sample_img) + c2 * et

        return sample_img
    
    @torch.no_grad
    def ddpm_sample(self, x_T):
        batch_size = x_T.shape[0]
        ddim_timestep_seq = np.asarray(list(range(0, self.time_steps)))
        ddim_timestep_next_seq = [-1] + list(ddim_timestep_seq[:-1])

        device = x_T.device
        # start from pure noise (for each example in the batch)
        sample_img = x_T
        for i in tqdm(reversed(range(self.time_steps)), desc='sampling loop time step', total=self.time_steps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), ddim_timestep_next_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = compute_alpha(self.betas, t)
            alpha_cumprod_t_next = compute_alpha(self.betas, next_t)
            
            beta_t = 1 - alpha_cumprod_t/alpha_cumprod_t_next
            output = self.model(sample_img, t)

            x0_from_e = (1.0/alpha_cumprod_t).sqrt() * sample_img - (1.0 / alpha_cumprod_t - 1).sqrt() * output
            x0_from_e = torch.clamp(x0_from_e, -1, 1)

            mean_eps = (
                (alpha_cumprod_t_next.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - alpha_cumprod_t_next)) * sample_img
            ) / (1.0 - alpha_cumprod_t)

            noise = torch.randn_like(sample_img)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample_img = mean_eps + mask * torch.exp(0.5 * logvar) * noise

        return sample_img

    def trainer(self, x):
        t = torch.randint(self.time_steps, size=(x.shape[0], ), device=x.device)
        noise = torch.randn_like(x, device=x.device)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise)
        pred = self.model(x_t, t)
        # loss_mse = F.mse_loss(pred, noise, reduction='mean')
        loss_l1 = F.l1_loss(pred, noise, reduction='sum')
        # return loss_mse + loss_l1*0.2
        return loss_l1

    def forward(self, x, mod='ddpm'):
        if self.training:
            return self.trainer(x)
        elif mod=='ddpm':
            return self.ddpm_sample(x)
        elif mod=='ddim':
            return self.ddim_sample(x)
    
if __name__ == "__main__":
    from models.backbone.unet import UNet, GhostUNet
    batch_size = 1
    time_steps = 1000
    x = torch.randn(batch_size, 3, 256, 256)
    model = GhostUNet(time_steps,dims=[64, 128, 128, 128, 128], num_blocks=[1, 2, 2, 2, 1])
    diffuser = GaussDiffuser(model.cuda().half(), 1e-3, 1e-2, time_steps)
    diffuser.cuda().half()
    x = x.cuda().half()
    diffuser.eval()
    loss = diffuser(x)

    print(loss.shape)