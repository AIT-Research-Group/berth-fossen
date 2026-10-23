# Berth Fossen

### Wind, reflected waves and probabilistic forecasting at marine berths

[![Tests](https://github.com/AIT-Research-Group/berth-fossen/actions/workflows/tests.yml/badge.svg)](https://github.com/AIT-Research-Group/berth-fossen/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Research software](https://img.shields.io/badge/status-research%20software-596579)](#scientific-scope)

A Python research framework for simulating berth wind, incident and reflected waves, and vessel dynamics using the Fossen three-degree-of-freedom formulation. The repository combines physics-based synthetic data generation with LSTM and conditional GAN-LSTM forecasting, probabilistic evaluation, and an interactive browser viewer.

Developed under [AIT Research Group](https://github.com/AIT-Research-Group).

**Start here:** [Installation](#installation) · [Quick start](#quick-start) · [Forecasting](#forecasting) · [Documentation](#documentation)

## Overview

The workflow supports simulation, causal feature extraction, scenario-based dataset preparation, model training and forecast evaluation. Environmental forecasts and direct load forecasts use separate input variables and checkpoints.

| Component | Implementation |
| :--- | :--- |
| Wind | Time-varying mean wind with Kaimal turbulence |
| Waves | JONSWAP spectrum, finite-depth dispersion and coherent linear reflection |
| Wave measurements | Synthetic probes with causal incident/reflected wave separation |
| Vessel dynamics | Planar surge, sway and yaw in the Fossen formulation |
| Reference hydrodynamics | Optional MSS tanker force RAOs and radiation-state reference data |
| Forecasting | Point LSTM, Energy-LSTM and conditional GAN-LSTM |
| Evaluation | Point errors, CRPS, interval coverage and event-probability metrics |
| Visualization | Local Plotly interface for waves, spectra, vessel loads and archived forecasts |

The framework is intended for reproducible numerical experiments. It has not been validated as an operational berthing-safety system.

## Installation

Requires **Python 3.10 or newer**. Clone the repository, create a virtual environment, then activate it before installing dependencies.

```bash
git clone https://github.com/AIT-Research-Group/berth-fossen.git
cd berth-fossen
python -m venv .venv
```

<details>
<summary><strong>Windows PowerShell</strong></summary>

```powershell
.venv/Scripts/Activate.ps1
python -m pip install ".[training,hydro,dev]"
```

</details>

<details>
<summary><strong>Linux and macOS</strong></summary>

```bash
source .venv/bin/activate
python -m pip install ".[training,hydro,dev]"
```

Use `python3` when your system does not provide the `python` command.

</details>

For a minimal NumPy-based simulation environment, install with `python -m pip install .`.

| Optional dependency group | Purpose |
| :--- | :--- |
| `training` | PyTorch forecasting and Matplotlib evaluation plots |
| `hydro` | SciPy support for the MSS tanker benchmark |
| `dev` | Test and packaging tools |
| `figures` | Publication figures and Playwright-based rendering |

The supplied training configurations request `"device": "cuda"`. Set this field to `"cpu"` in the configuration you use if no compatible GPU is available. A CUDA-enabled PyTorch installation must match the local system environment.

## Quick start

Run all commands from the repository root with the virtual environment activated.

### 1. Generate a demonstration simulation

```bash
python -m berth_sim --config configs/demo.json --output outputs/demo
```

The default demo generates 30 minutes of observations at 0.25-second intervals: 7,200 samples and 21 feature rows. This short run demonstrates the simulation pipeline; it does not provide enough history for the default 60-step forecasting models.

### 2. Open the interactive viewer

```bash
python scripts/wave_viewer.py
```

Open [localhost:8090](http://127.0.0.1:8090). The viewer reads saved simulation and experiment outputs and provides views of the wave surface, spectra, sea state, vessel response and available forecasts. It does not train models.

### 3. Run the test suite

```bash
python -m unittest discover -s tests -v
```

Tests that depend on saved experiments are skipped when those artifacts are absent. Generating the demo first enables the corresponding wave-viewer data tests.

**Output directories:** simulation and training commands require a new output directory. Use a different name for each run to preserve existing experiment results.

## Forecasting

### Models and forecast targets

| Model | Role |
| :--- | :--- |
| `lstm_point` | Deterministic LSTM baseline |
| `energy_lstm` | Ensemble generator trained with an energy-score objective |
| `gan_lstm` | Conditional ensemble generator with adversarial training |

With the supplied configurations, each model uses **60 minutes of history** to predict the next **30 minutes** at one-minute resolution.

| Task | Historical inputs and forecast targets |
| :--- | :--- |
| Environment | Two mean-wind components, wind gust and estimated reflected-wave Hm0 |
| Loads | RMS and absolute-maximum statistics for surge force, sway force and yaw moment |

Direct load forecasting uses historical simulated load statistics. It does not propagate environmental forecast ensembles through the vessel model. Environmental and load checkpoints are not interchangeable.

### Train an environmental forecasting experiment

```bash
python -m berth_sim.training.run --config configs/training.json --output outputs/training_run
```

The runner prepares the scenario dataset, trains the models and writes evaluation artifacts. The base configuration defines 56 four-hour scenarios, split into 32 training, 8 validation, 8 calibration and 8 test scenarios. Scenarios are separated before overlapping history/forecast windows are extracted.

### Predict from a trained environmental model

```bash
python -m berth_sim.training.predict --checkpoint outputs/training_run/gan_lstm/best.pt --features outputs/training_run/dataset/scenario_048.csv --calibrators outputs/training_run/evaluation/calibrators.json --output outputs/prediction
```

The example uses a scenario CSV generated by the preceding training run. The input must contain a complete, finite history of consecutive one-minute observations with valid wave features. Prediction uses the final history window and writes `forecast.csv`, `forecast.npz` and `events.json`.

### Train and use a direct load model

```bash
python -m berth_sim.training.force_run --config configs/force_training.json --output outputs/force_training_run
python -m berth_sim.training.force_predict --checkpoint outputs/force_training_run/gan_lstm/best.pt --input outputs/force_training_run/dataset/scenario_048.csv --output outputs/load_prediction
```

Load histories must satisfy `absolute maximum >= RMS >= 0`. Predictions retain these nonnegative statistical constraints; they do not represent instantaneous signed forces.

### Experiment configurations

| Configuration | Purpose |
| :--- | :--- |
| [`demo.json`](configs/demo.json) | Short physics simulation and viewer demonstration |
| [`training.json`](configs/training.json) | Environmental forecasting experiment |
| [`force_training.json`](configs/force_training.json) | Direct load forecasting with MSS reference data |
| [`training_v3_base.json`](configs/training_v3_base.json) | Larger baseline comparison experiment |
| [`training_v3_upgraded.json`](configs/training_v3_upgraded.json) | Larger experiment with forecasting upgrades |

Historical datasets, trained checkpoints and paper results are not included. Generate these artifacts with the relevant configuration before running downstream analysis or publication scripts.

## Generated artifacts

| Stage | Main outputs |
| :--- | :--- |
| Simulation | `observations.npz`, `truth.npz`, `features.csv`, `feature_truth.csv`, `vessel.npz`, `load_features.csv`, `metadata.json` |
| Dataset preparation | Scenario CSV files, split archives and a dataset manifest |
| Training | Model checkpoints, loss histories, scaler parameters and experiment records |
| Evaluation | Metric tables, forecast archives, calibration files, plots and reports |
| Environmental prediction | Forecast table, ensemble archive and event probabilities |
| Load prediction | Forecast table, ensemble archive and prediction metadata |

Simulation truth is retained for evaluation and kept separate from model input features. Vessel and load artifacts are generated when vessel simulation is enabled. Generated files remain under `outputs/`, which is excluded from version control.

## Repository structure

```text
berth-fossen/
├── configs/                 Simulation and training configurations
├── data/reference/          MSS tanker benchmark and upstream license
├── docs/                    Methods, conventions and validation notes
├── scripts/                 Viewers, analysis and publication utilities
├── src/berth_sim/            Simulation, dynamics, forecasting and evaluation
├── tests/                   Physics, forecasting and viewer data tests
├── viewer/                  Browser interface and bundled Plotly
├── .github/workflows/       Automated tests
└── pyproject.toml           Package metadata and dependency groups
```

## Scientific scope

The simulation uses synthetic weather scenarios, linear wave reflection and illustrative vessel parameters unless reference hydrodynamics are selected. The MSS tanker data describe open-water conditions; berth-wall effects and vessel–quay gap resonance are not resolved by those data.

Demonstration thresholds are experimental. Forecast metrics and event probabilities from the synthetic datasets do not establish field performance or operational safety. Coordinate conventions, model assumptions and calibration procedures are documented in the [model notes](docs/model-notes.md).

## Documentation

| Resource | Contents |
| :--- | :--- |
| [Model and experiment notes](docs/model-notes.md) | Physics, coordinate conventions, feature definitions, training methods and references |
| [Validation record](docs/validation.md) | Local verification, skipped checks and artifact availability |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | MSS and Plotly attribution |
| [GitHub Actions](https://github.com/AIT-Research-Group/berth-fossen/actions) | Automated test runs |

Publication utilities require saved experiment outputs. Install the `figures` dependency group for figure generation. The optional Word report utility additionally requires Node.js and `npm install` inside `scripts/paper_docx/`.

The current interface is served by `scripts/wave_viewer.py`. The legacy `serve_plots.py` API and its browser smoke scripts reference an older frontend that is not distributed here.

## Attribution and licensing

The MSS reference files retain their upstream MIT license in [`data/reference/mss_tanker/LICENSE`](data/reference/mss_tanker/LICENSE). The bundled Plotly library retains its copyright and MIT notice. Details are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

A license for the original project code has not yet been specified. Public availability does not grant additional reuse rights; contact the rights holder before redistributing or reusing that code.
