"""CPU-friendly training, validation-only checkpoint selection."""
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import Generator, Discriminator, energy_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(config):
    """Select the configured device with deterministic kernels; fail rather than fall back."""
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests CUDA but this PyTorch build/GPU is unavailable")
        # Required by deterministic cuBLAS (and hence LSTM) kernels; must precede CUDA use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    return device


@torch.no_grad()
def validation_score(model, data, samples, batch_size, seed, fair=False):
    model.eval()
    device = next(model.parameters()).device
    total = 0.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, len(data["x"]), batch_size):
            x = torch.as_tensor(data["x"][start:start+batch_size], device=device)
            y = torch.as_tensor(data["y"][start:start+batch_size], device=device)
            prediction = model(x, samples)
            p = model.transform.score_space(prediction)
            target = model.transform.score_space(y)
            score = energy_score(p, target, fair) if model.noise_size else (p[:, 0] - target).square().mean()
            total += score.item() * len(x)
    return total / len(data["x"])


def train_model(name, config, scaler, training, validation, directory):
    if name not in ("lstm_point", "energy_lstm", "gan_lstm"):
        raise ValueError(name)
    directory = Path(directory) / name
    directory.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    device = torch.device(config["device"])
    generator_class, discriminator_class = model_classes(config)
    model = generator_class(config, scaler, stochastic=name != "lstm_point").to(device)
    discriminator = discriminator_class(config, scaler).to(device) if name == "gan_lstm" else None
    # AdamW with weight_decay=0 is plain Adam, so older configs train exactly as before.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.))
    # Two time-scale update rule: the critic may learn faster than the generator (Heusel et al. 2017).
    critic_lr = config["learning_rate"] * config.get("discriminator_lr_ratio", 1.)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=critic_lr, betas=(.5, .999)) if discriminator else None
    loader_rng = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(training["x"]), torch.from_numpy(training["y"])),
                        batch_size=config["batch_size"], shuffle=True, generator=loader_rng)
    bce = nn.BCEWithLogitsLoss()
    hinge = config.get("adversarial_loss", "bce") == "hinge"
    critic_samples = config.get("critic_samples", 1)
    fair = config.get("fair_energy_score", False)
    best, stale = float("inf"), 0
    history = []
    started = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals = np.zeros(4)
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            samples = config["ensemble_train"] if model.noise_size else 1
            prediction = model(x, samples)
            distribution_loss = energy_score(model.transform.score_space(prediction), model.transform.score_space(y), fair) if model.noise_size else prediction.new_zeros(())
            mean_loss = (model.transform.score_space(prediction.mean(dim=1)) - model.transform.score_space(y)).square().mean()
            adversarial = prediction.new_zeros(())
            d_loss = prediction.new_zeros(())
            if discriminator:
                discriminator.train()
                optimizer_d.zero_grad(set_to_none=True)
                # Score several ensemble members, not only the first, so every member must look realistic.
                k = min(critic_samples, prediction.shape[1])
                histories = x.repeat_interleave(k, dim=0)
                members = prediction[:, :k].reshape(-1, *prediction.shape[2:])
                real = discriminator(x, y)
                fake = discriminator(histories, members.detach())
                if hinge:
                    d_loss = (nn.functional.relu(1 - real).mean() + nn.functional.relu(1 + fake).mean()) / 2
                else:
                    d_loss = (bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))) / 2
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.)
                optimizer_d.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                fake_for_generator = discriminator(histories, members)
                adversarial = -fake_for_generator.mean() if hinge else bce(fake_for_generator, torch.ones_like(fake_for_generator))
            weight = config["adversarial_weight"] if epoch > config["adversarial_warmup_epochs"] else 0.
            loss = (distribution_loss + config["mean_loss_weight"] * mean_loss + weight * adversarial) if model.noise_size else mean_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {name}, epoch={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if discriminator:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            totals += np.array([loss.item(), distribution_loss.item(), mean_loss.item(), d_loss.item()]) * len(x)
            count += len(x)
        score = validation_score(model, validation, config["ensemble_validation"] if model.noise_size else 1,
                                 config["batch_size"], config["seed"] + 1000, fair)
        row = dict(epoch=epoch, train_loss=totals[0]/count, energy_score=totals[1]/count,
                   mean_loss=totals[2]/count, discriminator_loss=totals[3]/count,
                   validation_score=score, elapsed_s=time.monotonic()-started)
        history.append(row)
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if score < best - 1e-6:
            best, stale = score, 0
            checkpoint = {"model_name": name, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "config": config, "scaler": scaler,
                "epoch": epoch, "validation_score": best, "synthetic_only": True,
                "torch_version": str(torch.__version__)}
            if discriminator:
                checkpoint["discriminator_state"] = discriminator.state_dict()
            torch.save(checkpoint, directory / "best.pt")
        else:
            stale += 1
        print(f"{name} epoch={epoch:02d} train={row['train_loss']:.4f} val={score:.4f} best={best:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    summary = {"epochs_run": len(history), "best_validation_score": best,
               "duration_s": time.monotonic()-started, "parameter_count": sum(p.numel() for p in model.parameters())}
    (directory / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator_class, _ = model_classes(checkpoint["config"])
    model = generator_class(checkpoint["config"], checkpoint["scaler"], checkpoint["model_name"] != "lstm_point").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def model_classes(config):
    if config.get("task") == "conditional_loads":
        from .conditional_model import ConditionalGenerator, ConditionalDiscriminator
        return ConditionalGenerator, ConditionalDiscriminator
    if config.get("task") == "loads":
        from .force_model import ForceGenerator, ForceDiscriminator
        return ForceGenerator, ForceDiscriminator
    return Generator, Discriminator


@torch.no_grad()
def predict(model, histories, samples=100, batch_size=64, seed=42):
    model.eval()
    device = next(model.parameters()).device
    if not model.noise_size:
        samples = 1
    results = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, len(histories), batch_size):
            x = torch.as_tensor(histories[start:start+batch_size], dtype=torch.float32, device=device)
            results.append(model(x, samples).cpu().numpy())
    return np.concatenate(results)
