
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

**Note on performance**  
Use the demo defaults (`n=128, K=16, frames=36`). Scale gradually for heavier experiments.
