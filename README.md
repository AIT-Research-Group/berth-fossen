# Berth wind and reflected-wave forecasting

Research simulator for berth wind, incident and reflected waves, and planar vessel dynamics. Includes point LSTM, Energy-LSTM and conditional GAN-LSTM forecasting, evaluation tools and a local Plotly viewer.

## Setup

Python 3.10 or newer is required. Run commands from the repository root.

```sh
python -m venv .venv
python -m pip install ".[training,hydro,dev]"
```

Activate the virtual environment before installing. On Windows PowerShell use `.venv/Scripts/Activate.ps1`; on Linux or macOS use `source .venv/bin/activate`. For simulation only, install with `python -m pip install .`. CUDA installations depend on the local PyTorch environment. Training configurations request CUDA by default; set `device` to `cpu` when using a CPU.

## Simulation and viewer

```sh
python -m berth_sim --config configs/demo.json --output outputs/demo
python scripts/wave_viewer.py
```

Open http://127.0.0.1:8090 after starting the viewer. Create simulation outputs first. Each simulation or training run requires a new output directory.

## Training and prediction

```sh
python -m berth_sim.training.run --config configs/training.json --output outputs/training_run
python -m berth_sim.training.predict --checkpoint outputs/training_run/gan_lstm/best.pt --features outputs/training_run/dataset/scenario_048.csv --calibrators outputs/training_run/evaluation/calibrators.json --output outputs/prediction
python -m berth_sim.training.force_run --config configs/force_training.json --output outputs/force_training_run
```

The prediction CSV must contain at least 60 consecutive valid feature rows. `configs/training_v3_base.json` and `configs/training_v3_upgraded.json` define the larger comparison experiments. Training creates datasets, checkpoints, evaluation tables and plots under the selected output directory. Historical checkpoints and experimental outputs are not distributed with this repository.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests requiring saved experiments skip when those outputs are absent. Run the simulation example first to exercise the demo replay tests. Browser checks require Playwright and a supported installed browser. The legacy `serve_plots.py` API and its browser smoke scripts refer to an older `web/` frontend that is absent from the supplied project; use `wave_viewer.py` for the current viewer.

## Repository contents

| Directory | Contents |
| :--- | :--- |
| `src/berth_sim/` | Physics, simulation, CLI, forecasting and evaluation |
| `configs/` | Demo and training configurations |
| `tests/` | Physics, learning and viewer data tests |
| `scripts/` | Viewers, analysis and publication utilities |
| `viewer/` | Current browser interface and bundled Plotly |
| `data/reference/mss_tanker/` | MSS reference tables and upstream license |
| `docs/model-notes.md` | Detailed original methods, conventions and experiment notes |

Publication utilities consume saved experiment outputs, which must be generated separately. For the optional Word report utility, run `npm install` in `scripts/paper_docx` before running `build.js`. Additional figure dependencies are available with `python -m pip install ".[figures]"`.

## Scientific scope

The model uses Kaimal wind, JONSWAP waves, linear reflection, causal wave separation and Fossen planar dynamics. Demonstration vessel coefficients are illustrative. The optional MSS tanker benchmark provides open-water reference hydrodynamics; it does not resolve berth-wall or gap resonance effects. Training uses synthetic scenarios. Thresholds are experimental and do not establish operational berthing safety. Environmental forecasting and direct load forecasting are separate tasks. See the detailed model notes for coordinate conventions, calibration and limitations.

## Third-party material and reuse

MSS reference files retain their upstream MIT license. Plotly retains its bundled copyright and MIT notice. See `THIRD_PARTY_NOTICES.md`. No license for the original project code was specified in the supplied source; permission to redistribute or reuse that code must be obtained from its rights holder.
