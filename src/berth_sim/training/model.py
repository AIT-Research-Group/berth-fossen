"""LSTM context, stochastic sequence decoder, wind and wave output branches."""
import numpy as np
import torch
from torch import nn


def physical_to_latent(values):
    values = np.asarray(values)
    gust_increment = np.maximum(values[..., 2] - np.linalg.norm(values[..., :2], axis=-1), 1e-4)
    energy = np.maximum((values[..., 3] / 4)**2, 1e-8)
    return np.stack((values[..., 0], values[..., 1], np.log(gust_increment), np.log(energy)), axis=-1)


def fit_scaler(unique_training_features):
    latent = physical_to_latent(unique_training_features)
    return {"latent_mean": latent.mean(axis=0).tolist(),
            "latent_std": np.maximum(latent.std(axis=0), .05).tolist(),
            "physical_mean": unique_training_features.mean(axis=0).tolist(),
            "physical_std": np.maximum(unique_training_features.std(axis=0), .01).tolist()}


class PhysicalTransform(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        for name, values in scaler.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))

    def encode(self, physical):
        delta = (physical[..., 2] - torch.linalg.vector_norm(physical[..., :2], dim=-1)).clamp_min(1e-4)
        energy = (physical[..., 3] / 4).square().clamp_min(1e-8)
        latent = torch.stack((physical[..., 0], physical[..., 1], delta.log(), energy.log()), dim=-1)
        return (latent - self.latent_mean) / self.latent_std

    def decode(self, standardized):
        latent = standardized * self.latent_std + self.latent_mean
        components = latent[..., :2]
        increment = latent[..., 2].clamp(-15, 10).exp()
        energy = latent[..., 3].clamp(-20, 6).exp()
        gust = torch.linalg.vector_norm(components, dim=-1) + increment
        hs = 4 * energy.sqrt()
        return torch.cat((components, gust.unsqueeze(-1), hs.unsqueeze(-1)), dim=-1)

    def score_space(self, physical):
        return (physical - self.physical_mean) / self.physical_std


class Generator(nn.Module):
    """Optional upgrades (all off by default, so older checkpoints load unchanged):

    ``dropout`` on the encoder context and decoder output, and ``local_noise_size``
    fresh noise per lead-time step, so ensemble members can diverge along the
    horizon instead of differing only by one global draw.
    """

    def __init__(self, config, scaler, stochastic=True):
        super().__init__()
        self.transform = PhysicalTransform(scaler)
        self.horizon = config["horizon_steps"]
        self.noise_size = config["noise_size"] if stochastic else 0
        self.local_noise_size = config.get("local_noise_size", 0) if stochastic else 0
        hidden = config["hidden_size"]
        self.encoder = nn.LSTM(4, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden + self.noise_size + self.local_noise_size + 1, hidden, batch_first=True)
        self.dropout = nn.Dropout(config.get("dropout", 0.))
        self.wind_head = nn.Linear(hidden, 3)
        self.wave_head = nn.Linear(hidden, 1)
        for head in (self.wind_head, self.wave_head):
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)

    def forward(self, history, samples=1, noise=None):
        standardized = self.transform.encode(history)
        _, (hidden, _) = self.encoder(standardized)
        batch = len(history)
        context = self.dropout(hidden[-1])[:, None, :].expand(-1, samples, -1)
        if noise is None:
            noise = torch.randn(batch, samples, self.noise_size, device=history.device)
        context = torch.cat((context, noise), dim=-1).reshape(batch * samples, 1, -1)
        timeline = torch.linspace(1/self.horizon, 1, self.horizon, device=history.device)[None, :, None]
        parts = [context.expand(-1, self.horizon, -1), timeline.expand(batch*samples, -1, -1)]
        if self.local_noise_size:
            parts.insert(1, torch.randn(batch*samples, self.horizon, self.local_noise_size, device=history.device))
        decoded, _ = self.decoder(torch.cat(parts, dim=-1))
        decoded = self.dropout(decoded)
        residual = torch.cat((self.wind_head(decoded), self.wave_head(decoded)), dim=-1)
        residual = residual.reshape(batch, samples, self.horizon, 4)
        # A residual around the last observation provides a persistence starting point.
        forecast = standardized[:, None, -1:, :] + residual
        return self.transform.decode(forecast)


def _normalized(layer, enabled):
    return nn.utils.parametrizations.spectral_norm(layer) if enabled else layer


class Discriminator(nn.Module):
    """Conditional critic on (60-min history, 30-min future).

    ``critic: "multiscale"`` looks at the future both minute by minute and as
    5-minute means (convolutions), instead of one flattened vector;
    ``spectral_norm`` bounds the critic's Lipschitz constant (Miyato et al. 2018).
    """

    def __init__(self, config, scaler):
        super().__init__()
        hidden = config["hidden_size"]
        norm = config.get("spectral_norm", False)
        self.multiscale = config.get("critic", "flat") == "multiscale"
        self.transform = PhysicalTransform(scaler)
        self.encoder = nn.LSTM(4, hidden, batch_first=True)
        if self.multiscale:
            self.fine = nn.Sequential(_normalized(nn.Conv1d(4, hidden, 3, padding=1), norm), nn.LeakyReLU(.2),
                                      _normalized(nn.Conv1d(hidden, hidden, 3, padding=1), norm), nn.LeakyReLU(.2))
            self.coarse = nn.Sequential(_normalized(nn.Conv1d(4, hidden, 3, padding=1), norm), nn.LeakyReLU(.2))
            self.classifier = nn.Sequential(_normalized(nn.Linear(3 * hidden, hidden), norm), nn.LeakyReLU(.2),
                                            _normalized(nn.Linear(hidden, 1), norm))
        else:
            self.classifier = nn.Sequential(_normalized(nn.Linear(hidden + config["horizon_steps"]*4, hidden), norm),
                                            nn.LeakyReLU(.2), _normalized(nn.Linear(hidden, 1), norm))

    def forward(self, history, future):
        _, (hidden, _) = self.encoder(self.transform.score_space(history))
        values = self.transform.score_space(future)
        if not self.multiscale:
            return self.classifier(torch.cat((hidden[-1], values.flatten(1)), dim=1)).squeeze(-1)
        values = values.transpose(1, 2)
        fine = self.fine(values).mean(-1)
        coarse = self.coarse(nn.functional.avg_pool1d(values, min(5, values.shape[-1]))).mean(-1)
        return self.classifier(torch.cat((hidden[-1], fine, coarse), dim=-1)).squeeze(-1)


def energy_score(ensemble, observation, fair=False):
    """Empirical multivariate ES, divided by sqrt(dimension) for scale.

    ensemble [B,M,H,D], observation [B,H,D]; score standardized physical variables.
    The plain estimator averages the spread over all M^2 pairs, including the M
    zero self-pairs, so it under-rewards spread by (M-1)/M and favours
    under-dispersed ensembles when M is small. ``fair=True`` averages over the
    M(M-1) distinct pairs (fair score, Ferro 2014).
    """
    vectors = ensemble.flatten(2)
    truth = observation.flatten(1)
    accuracy = torch.linalg.vector_norm(vectors - truth[:, None], dim=-1).mean(dim=1)
    members = vectors.shape[1]
    pairs = torch.cdist(vectors, vectors).sum(dim=(1, 2))
    diversity = pairs / (members * (members - 1) if fair and members > 1 else members**2) / 2
    return ((accuracy - diversity) / np.sqrt(vectors.shape[-1])).mean()
