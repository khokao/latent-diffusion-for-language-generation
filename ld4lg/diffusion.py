import math
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput
from accelerate.logging import get_logger

from collections import defaultdict

logger = get_logger(__name__)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        transformer,
        autoencoder,
        seq_len,
        x_dim,
        timesteps,
        beta_schedule,
        objective,
        loss_type,
        self_condition_prob,
        class_uncondition_prob,
        num_classes,
        length_distribution,
        tokenizer,
    ):
        super().__init__()
        self.transformer = transformer
        self.autoencoder = autoencoder
        self.seq_len = seq_len
        self.x_dim = x_dim
        self.timesteps = timesteps
        self.objective = objective
        self.self_condition_prob = self_condition_prob
        self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=class_uncondition_prob)
        self.num_classes = num_classes
        self.length_distribution = length_distribution
        self.tokenizer = tokenizer

        if loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f'Invalid loss type: {loss_type}')

        self._init_diffusion_params(timesteps, beta_schedule)
        self._init_normalize_params(x_dim)

    def forward(self, input_ids, attention_mask, class_id=None):
        with torch.no_grad():
            x0 = self.autoencoder.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

            batch_size, seq_len, x_dim = x0.shape

            if not self.norm_initialized:
                self._init_normalize_params(x_dim=x_dim, x=x0, x_mask=attention_mask)

            x0 = self.normalize(x0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        x_self_cond = None
        if self.transformer.use_self_condition and random.random() < self.self_condition_prob:
            with torch.no_grad():
                tmp_preds = self.get_diffusion_model_preds(xt, x_mask=attention_mask, t=t, class_id=class_id)
                x_self_cond = tmp_preds['pred_x0'].detach()

        if self.transformer.use_class_condition and self.class_unconditional_bernoulli.probs > 0:
            class_unconditional_mask = self.class_unconditional_bernoulli.sample((batch_size,)).bool()
            class_id[class_unconditional_mask] = self.num_classes

        preds = self.get_diffusion_model_preds(
            xt,
            x_mask=attention_mask,
            t=t,
            x_self_cond=x_self_cond,
            class_id=class_id,
        )

        if self.objective == 'pred_noise':
            loss = self.criterion(preds['pred_noise'], noise)
        elif self.objective == 'pred_x0':
            loss = self.criterion(preds['pred_x0'], x0)
        else:
            raise ValueError(f'Invalid objective: {self.objective}')

        return loss

    def get_diffusion_model_preds(self, xt, x_mask, t, x_self_cond=None, class_id=None):
        out = self.transformer(xt, x_mask=x_mask, t=t, x_self_cond=x_self_cond, class_id=class_id)

        if self.objective == 'pred_noise':
            pred_noise = out
            pred_x0 = self.p_sample_x0(xt, t, pred_noise)
        elif self.objective == 'pred_x0':
            pred_x0 = out
            pred_noise = self.p_sample_noise(xt, t, pred_x0)
        else:
            raise ValueError(f'Invalid objective: {self.objective}')

        preds = {
            'pred_noise': pred_noise,
            'pred_x0': pred_x0,
        }
        return preds

    def q_sample(self, x0, t, noise):
        xt = (
            self.extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        return xt

    def p_sample_x0(self, xt, t, noise):
        x0 = (
            self.extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
        )
        return x0

    def p_sample_noise(self, xt, t, x0):
        noise = (
            (self.extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt - x0)
            / self.extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape)
        )
        return noise

    def normalize(self, x, eps=1e-5):
        x = (x - self.x_mean) / (self.x_std + eps)
        return x

    def unnormalize(self, x, eps=1e-5):
        x = x * (self.x_std + eps) + self.x_mean
        return x

    def _init_diffusion_params(self, timesteps, beta_schedule):
        betas = self.get_betas(timesteps, beta_schedule)
        alphas = 1. - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))  # NOQA
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance', posterior_log_variance)
        register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def _init_normalize_params(self, x_dim, x=None, x_mask=None):
        if x is None and x_mask is None:
            mean = torch.zeros(x_dim)
            std = torch.ones(1)
            self.norm_initialized = False
        elif x is not None and x_mask is not None:
            tmp_x = [x[i][:torch.sum(x_mask[i])] for i in range(x.shape[0])]
            tmp_x = torch.cat(tmp_x, dim=0)

            mean = torch.mean(tmp_x, dim=0)
            std = torch.std(tmp_x - mean, unbiased=False)
            self.norm_initialized = True
        else:
            raise ValueError('x and x_mask must be both None or not None')

        assert mean.shape == (x_dim,)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))  # NOQA
        register_buffer('x_mean', mean)
        register_buffer('x_std', std)

    @staticmethod
    def get_betas(timesteps, schedule):
        if schedule == 'linear':
            start = 0.0001
            end = 0.02
            betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
        elif schedule == 'cosine':
            s = 0.008
            max_beta = 0.999

            def func(t, timesteps=timesteps, s=s):
                numer = (t / timesteps + s) * math.pi
                denom = (1 + s) * 2
                return math.cos(numer / denom) ** 2

            betas = []
            for t in range(timesteps):
                beta_t = min(1 - func(t + 1) / func(t), max_beta)
                betas.append(beta_t)
            betas = torch.tensor(betas, dtype=torch.float64)
        elif schedule == 'sqrt':
            s = 0.0001
            max_beta = 0.999

            def func(t, timesteps=timesteps, s=s):
                return 1 - math.sqrt(t / timesteps + s)

            betas = []
            for t in range(timesteps):
                beta_t = min(1 - func(t + 1) / func(t), max_beta)
                betas.append(beta_t)
            betas = torch.tensor(betas, dtype=torch.float64)
        else:
            raise NotImplementedError(f'Unknown beta schedule: {schedule}')

        return betas

    @staticmethod
    def extract(input, indices, shape):
        output = input.gather(-1, indices)
        batch_size, *_ = output.shape
        output = output.reshape(batch_size, *((1,) * (len(shape) - 1)))
        return output

    @torch.inference_mode()
    def sample(self, num_samples=25, batch_size=32, class_id=None, sampling_steps=250, strategy=None, eta=0.0):
        if strategy is None:
            logger.warning('No strategy provided, using greedy strategy WITH DEFAULT PARAMS')
            strategy = {'greedy': {}}

        outputs = defaultdict(list)
        num_sampled = 0
        while num_sampled < num_samples:
            latent, x_mask = self.sample_ddim(batch_size, class_id, sampling_steps, eta)

            for strategy_name, strategy_kwargs in strategy.items():
                output_texts = self.sample_decode(self.tokenizer, latent, x_mask, strategy_kwargs)
                outputs[strategy_name].extend(output_texts)

            num_sampled += batch_size
        outputs = dict(outputs)

        return outputs

    @torch.inference_mode()
    def sample_ddim(self, batch_size=32, class_id=None, sampling_steps=250, eta=0.0):
        lengths = self.length_distribution.sample((batch_size,)).tolist()

        t = torch.linspace(-1, self.timesteps - 1, steps=sampling_steps + 1)
        t = list(reversed(t.int().tolist()))
        t_pairs = list(zip(t[:-1], t[1:]))

        device = self.betas.device
        latent = torch.randn(
            (batch_size, self.seq_len, self.x_dim),
            device=device,
        )
        x_mask = torch.tensor(
            [[True] * length + [False] * (self.seq_len - length) for length in lengths],
            dtype=bool,
            device=device,
        )

        x_self_cond = None
        for t_prev, t_next in tqdm(t_pairs, desc='denoising...'):
            t_cond = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

            preds = self.get_diffusion_model_preds(
                xt=latent,
                x_mask=x_mask,
                t=t_cond,
                x_self_cond=x_self_cond,
                class_id=class_id,
            )

            if t_next < 0:
                latent = preds['pred_x0']
                break

            if self.transformer.use_self_condition:
                x_self_cond = preds['pred_x0']

            alpha = self.alphas_cumprod(t_prev)
            alpha_next = self.alphas_cumprod(t_next)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()

            latent = (
                alpha_next.sqrt() * preds['pred_x0']
                + (1 - alpha_next - sigma ** 2).sqrt() * preds['pred_noise']
                + torch.randn_like(latent) * sigma
            )

        latent = self.unnormalize(latent)

        return latent, x_mask

    @torch.inference_mode()
    def sample_decode(self, latent, x_mask, strategy_kwargs):
        encoder_outputs = BaseModelOutput(last_hidden_state=latent.clone())

        output_ids = self.autoencoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=x_mask.clone(),
            **strategy_kwargs,
        )

        output_texts = [
            self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for seq in output_ids
        ]
        output_texts = [text.strip() for text in output_texts]

        return output_texts
