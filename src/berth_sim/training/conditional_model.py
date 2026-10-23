"""Environmental-context load forecaster with signed means and local noise."""
import numpy as np
import torch
from torch import nn

TARGETS = [f"environment_{axis}_{stat}_{unit}" for stat in ("mean", "rms", "absmax")
           for axis, unit in (("X", "N"), ("Y", "N"), ("N", "Nm"))]
CONTEXT = ["wind_x_mps", "wind_y_mps", "wind_gust_mps", "wave_hs_m", "wave_tp_s",
           "wave_direction_sin", "wave_direction_cos", "displacement_x_m",
           "displacement_y_m", "yaw_rad", "u_mps", "v_mps", "r_radps"]
INPUTS = TARGETS + CONTEXT


def fit_scaler(rows):
    target = rows[:, :9]
    return {"input_mean": rows.mean(0).tolist(),
            "input_std": np.maximum(rows.std(0), 1e-5).tolist(),
            "physical_mean": target.mean(0).tolist(),
            "physical_std": np.maximum(target.std(0), 1.).tolist(),
            "axis_scale": np.maximum(np.sqrt((target[:, :3]**2).mean(0)), 1.).tolist()}


class LoadTransform(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        for key, value in scaler.items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))

    def history_space(self, values):
        return (values-self.input_mean)/self.input_std

    def score_space(self, values):
        return (values-self.physical_mean)/self.physical_std

    def encode(self, values):
        mean = values[..., :3]/self.axis_scale
        rms_gap = (values[..., 3:6]-values[..., :3].abs())/self.axis_scale
        max_gap = (values[..., 6:9]-values[..., 3:6])/self.axis_scale
        return torch.cat((mean, rms_gap.clamp_min(1e-5).log(), max_gap.clamp_min(1e-5).log()), -1)

    def decode(self, values):
        mean = values[..., :3]*self.axis_scale
        rms = mean.abs()+values[..., 3:6].clamp(-15, 10).exp()*self.axis_scale
        maximum = rms+values[..., 6:9].clamp(-15, 10).exp()*self.axis_scale
        return torch.cat((mean, rms, maximum), -1)


class ConditionalGenerator(nn.Module):
    def __init__(self, config, scaler, stochastic=True):
        super().__init__()
        self.transform = LoadTransform(scaler)
        self.horizon = config["horizon_steps"]
        self.noise_size = config["noise_size"] if stochastic else 0
        self.local_noise_size = config.get("local_noise_size", 4) if stochastic else 0
        hidden = config["hidden_size"]
        self.encoder = nn.LSTM(len(INPUTS), hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden+self.noise_size+self.local_noise_size+1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 9)
        nn.init.normal_(self.head.weight, std=.01); nn.init.zeros_(self.head.bias)

    def forward(self, history, samples=1):
        _, (hidden, _) = self.encoder(self.transform.history_space(history))
        batch = len(history)
        noise = torch.randn(batch, samples, self.noise_size, device=history.device)
        context = torch.cat((hidden[-1][:, None].expand(-1, samples, -1), noise), -1)
        context = context.reshape(batch*samples, 1, -1).expand(-1, self.horizon, -1)
        local = torch.randn(batch*samples, self.horizon, self.local_noise_size, device=history.device)
        steps = torch.linspace(1/self.horizon, 1, self.horizon, device=history.device)[None, :, None].expand(batch*samples, -1, -1)
        decoded, _ = self.decoder(torch.cat((context, local, steps), -1))
        residual = self.head(decoded).reshape(batch, samples, self.horizon, 9)
        return self.transform.decode(self.transform.encode(history[:, None, -1:, :9])+residual)


class ConditionalDiscriminator(nn.Module):
    """Conditional critic viewing individual minutes and five-minute averages."""
    def __init__(self, config, scaler):
        super().__init__()
        self.transform = LoadTransform(scaler)
        hidden = config["hidden_size"]
        self.encoder = nn.LSTM(len(INPUTS), hidden, batch_first=True)
        self.fine = nn.Sequential(nn.Conv1d(9, hidden, 3, padding=1), nn.LeakyReLU(.2),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.LeakyReLU(.2))
        self.coarse = nn.Sequential(nn.Conv1d(9, hidden, 3, padding=1), nn.LeakyReLU(.2))
        self.head = nn.Sequential(nn.Linear(3*hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, 1))

    def forward(self, history, future):
        _, (hidden, _) = self.encoder(self.transform.history_space(history))
        values = self.transform.score_space(future).transpose(1, 2)
        fine = self.fine(values).mean(-1)
        coarse = self.coarse(torch.nn.functional.avg_pool1d(values, min(5, values.shape[-1]))).mean(-1)
        return self.head(torch.cat((hidden[-1], fine, coarse), -1)).squeeze(-1)
