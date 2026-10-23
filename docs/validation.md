# Repository validation

Validated on 2026-10-23 with Python 3.11 and the existing project environment.

* Demo simulation: 7,200 samples, 21 feature rows, all feature rows valid.
* Unit tests after generating the demo: 54 discovered, 51 passed, 3 skipped.
* Skipped tests require saved force-training outputs, a saved load replay, or forecast archives.
* Python wheel and source archive built successfully using setuptools.
* Browser rendering, full model retraining and the GitHub Actions matrix were not executed.

The catalog test now creates temporary environmental and load CSV fixtures, so a fresh clone does not depend on local experiment results. Original simulation and forecasting implementations are retained.

Generated outputs, caches, virtual environments, historical paper revisions and personal working documents are excluded from the handoff. The detailed model notes retain historical experiment paths for reproducibility; those experiment artifacts are not bundled.
