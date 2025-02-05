import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mygru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, input_dim, hidden_dim):
        super().__init__()
        
        this_layer = n_layer
        self.g_ir = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_iz = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_in = funcsgru(this_layer, input_dim, hidden_dim, 0)
        self.g_hr = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hz = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.g_hn = funcsgru(this_layer, hidden_dim, hidden_dim, 0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        r_t = self.sigmoid(
            self.g_ir(x) + self.g_hr(h)
        )
        z_t = self.sigmoid(
            self.g_iz(x) + self.g_hz(h)
        )
        n_t = self.tanh(
            self.g_in(x) + self.g_hn(h).mul(r_t)
        )
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t

class funcsgru(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class funcs(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class matrix_vec_light(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x_len = x.size()[-2]
        x = x.squeeze(1).split([1, x_len - 1], dim = -2)[0].squeeze(1)
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))

class matrix_to_vec(nn.Module):
    def __init__(self, dim, p_num, in_channels = 1, out_channels = 30, vec_dim = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding = 1)
        self.conv2_drop = nn.Dropout2d()
        self.node_dim = vec_dim
        self.fc2 = nn.Linear(int(self.node_dim / 4) * int(p_num / 4), dim)

    def forward(self, x):
        bs, _, p_num, _ = x.size()
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # [6, 10, 1, 159]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(bs, int(self.node_dim / 4) * int(p_num / 4))
        return self.fc2(x)

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

class CondGaussianDiffusionX0N:
    '''与model.CondGaussianDiffusionX0N， 但sample过程带梯度'''
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='cosine'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = self.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.mse = torch.nn.MSELoss(reduce = False, reduction = 'none')
    
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        device = t.device
        out = a.to(t.device).gather(0, t).float() 
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out.to(device)
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        x_recon = model(x_t, t)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    # @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    # @torch.no_grad()
    def p_sample_loop(self, model, condition):
        batch_size, channel, ques_num, dim = condition.size()
        device = next(model.parameters()).device
        x_t = torch.randn(batch_size, channel, ques_num, int(dim / 3)).to(device)
        for i in reversed(range(0, self.timesteps)):
            in_x_t =model.x_cond_map(torch.cat([x_t, condition], dim = -1))
            x_t = self.p_sample(model, in_x_t, torch.full((batch_size,), i, device=device, dtype=torch.long))#.detach()
        return x_t
   
    def sample(self, model, x_t, channels=3):
        return self.p_sample_loop(model, x_t)
    
    # compute train losses
    def train_losses(self, model, x_start, condition):
        # generate random noise
        bs = x_start.size()[0]
        t = torch.randint(0, self.timesteps, (bs,)).to(x_start.device)
        noise = torch.randn_like(x_start).to(x_start.device)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        model_in = model.x_cond_map(torch.cat([condition, x_noisy], dim = -1))
        predicted_x0 = model(model_in, t)
        predict_num = x_start.size()[2] - 1
        loss = self.mse(x_start.split([1, predict_num], dim = -2)[0], 
                          predicted_x0.split([1, predict_num], dim = -2)[0])
        return loss

class CondGaussianDiffusionX0N3:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='cosine',
        x_t_dim = 64
    ):
        self.timesteps = timesteps
        self.x_t_dim = x_t_dim
        
        if beta_schedule == 'linear':
            betas = self.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.mse = torch.nn.MSELoss(reduce = False, reduction = 'none')
    
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        device = t.device
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out.to(device)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        x_recon = model(x_t, t)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    # @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    # @torch.no_grad()
    def p_sample_loop(self, model, condition):
        batch_size, channel, ques_num, dim = condition.size()
        device = next(model.parameters()).device
        x_t = torch.randn(batch_size, channel, ques_num, self.x_t_dim).to(device)
        for i in reversed(range(0, self.timesteps)):
            in_x_t =model.x_cond_map(torch.cat([x_t, condition], dim = -1))
            x_t = self.p_sample(model, in_x_t, torch.full((batch_size,), i, device=device, dtype=torch.long))#.detach()
        return x_t
   
    def sample(self, model, x_t, channels=3):
        return self.p_sample_loop(model, x_t)
    
    # compute train losses
    def train_losses(self, model, x_start, condition):
        bs = x_start.size()[0]
        t = torch.randint(0, self.timesteps, (bs,)).to(x_start.device)
        noise = torch.randn_like(x_start).to(x_start.device)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        model_in = model.x_cond_map(torch.cat([condition, x_noisy], dim = -1))
        predicted_x0 = model(model_in, t)
        predict_num = x_start.size()[2] - 1
        loss = self.mse(x_start.split([1, predict_num], dim = -2)[0], 
                          predicted_x0.split([1, predict_num], dim = -2)[0])
        return loss

class mini_resnet(nn.Module):
    '''
    classifier decoder implemented with mlp
    '''
    def __init__(self, n_layer, hidden_dim, out_dim, dpo):
        super().__init__()

        self.x_cond_map = nn.Linear(hidden_dim, out_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(out_dim),
            nn.Linear(out_dim, out_dim * 2),
            nn.Mish(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.mid_layer = nn.Linear(2 * out_dim, out_dim)
        self.lins = nn.ModuleList([
            nn.Linear(out_dim, out_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p = dpo)
        self.out = nn.Linear(out_dim, out_dim)
        self.norm= nn.LayerNorm(out_dim)

    def forward(self, x, t):
        bs, _, n, dim = x.size()
        x = x.reshape(bs, n, dim)
        t_emb = self.time_mlp(t).unsqueeze(1).repeat(1, n, 1)
        x = torch.cat([x, t_emb], dim = -1)
        x = F.relu(self.mid_layer(x))
        for lin in self.lins:
            x = self.norm(F.relu(lin(x)) + x)
        return self.out(self.dropout(x)).unsqueeze(1)

