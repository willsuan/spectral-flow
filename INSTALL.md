
# Installation & Troubleshooting

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Quick demo that produces figures
```bash
python scripts/run_quick_demo.py
```

**Common issues**
- `ModuleNotFoundError: spectral_flow_research` → run `pip install -e .` in the same environment used by your terminal or notebook.
- `scipy` missing or build errors → upgrade pip (`pip install --upgrade pip`), ensure Python 3.10/3.11, then `pip install scipy`.
- No figures after running the notebook → the first cell auto‑installs; ensure the kernel is the same environment.

**Note on performance**  
Use the demo defaults (`n=128, K=16, frames=36`). Scale gradually for heavier experiments.
