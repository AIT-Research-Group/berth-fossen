"""Forecast total environmental load RMS and absolute maxima, in SI units."""
import numpy as np
import torch
from torch import nn

TARGETS = [f"environment_{axis}_{stat}_{unit}" for stat in ("rms", "absmax")
           for axis, unit in (("X", "N"), ("Y", "N"), ("N", "Nm"))]


def latent(values):
    return np.log(np.maximum(np.concatenate((values[..., :3], values[..., 3:]-values[..., :3]), axis=-1), 1e-3))


def fit_force_scaler(training_rows):
    transformed = latent(training_rows)
    return {"latent_mean": transformed.mean(0).tolist(),
            "latent_std": np.maximum(transformed.std(0), .05).tolist(),
            "physical_mean": training_rows.mean(0).tolist(),
            "physical_std": np.maximum(training_rows.std(0), 1.).tolist()}


class ForceTransform(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        for key, values in scaler.items():
            self.register_buffer(key, torch.tensor(values, dtype=torch.float32))

    def encode(self, x):
        transformed = torch.cat((x[..., :3], x[..., 3:]-x[..., :3]), dim=-1).clamp_min(1e-3).log()
        return (transformed-self.latent_mean)/self.latent_std

    def decode(self, x):
        values = (x*self.latent_std+self.latent_mean).clamp(-15, 30).exp()
        return torch.cat((values[..., :3], values[..., :3]+values[..., 3:]), dim=-1)

    def score_space(self, x):
        return (x-self.physical_mean)/self.physical_std


class ForceGenerator(nn.Module):
    def __init__(self, config, scaler, stochastic=True):
        super().__init__()
        self.transform = ForceTransform(scaler)
        self.horizon = config['horizon_steps']
        self.noise_size = config['noise_size'] if stochastic else 0
        hidden = config['hidden_size']
        self.encoder = nn.LSTM(6, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden+self.noise_size+1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 6)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, samples=1):
        x = self.transform.encode(history)
        _, (hidden, _) = self.encoder(x)
        batch = len(x)
        noise = torch.randn(batch, samples, self.noise_size, device=x.device)
        context = torch.cat((hidden[-1][:, None].expand(-1, samples, -1), noise), dim=-1)
        context = context.reshape(batch*samples, 1, -1).expand(-1, self.horizon, -1)
        steps = torch.linspace(1/self.horizon, 1, self.horizon, device=x.device)[None, :, None].expand(batch*samples, -1, -1)
        out, _ = self.decoder(torch.cat((context, steps), dim=-1))
        residual = self.head(out).reshape(batch, samples, self.horizon, 6)
        return self.transform.decode(x[:, None, -1:, :]+residual)


class ForceDiscriminator(nn.Module):
    def __init__(self, config, scaler):
        super().__init__()
        self.transform = ForceTransform(scaler)
        hidden = config['hidden_size']
        self.encoder = nn.LSTM(6, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden+config['horizon_steps']*6, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, 1))

    def forward(self, history, future):
        _, (hidden, _) = self.encoder(self.transform.score_space(history))
        return self.head(torch.cat((hidden[-1], self.transform.score_space(future).flatten(1)), dim=-1)).squeeze(-1)
