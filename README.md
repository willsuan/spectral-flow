
# Spectral Flow Kit

A numerical package for studying and visualizing **spectral flow** and **harmonic transport** of the Dirichlet Laplacian on **evolving 2D domains**.

What you get:
- **Core pipeline**: evolving masks → Laplacians → eigen-tracking → connections/curvature → holonomy → rich figures.
- **Gauge choices**: overlap vs **log‑polar** (Lie‑algebra) connection with **symmetric curvature**.
- **Hadamard** shape derivative (fixed/adaptive δ‑band) + **timing profiles** for boundary velocity/acceleration.
- **Equal‑area loops**, **shape‑space metrics** & **geodesic** paths.
- **Ablations**, **convergence probes**, **sanity checks**, **repro manifests**, **figure stamps**.
- **Notebook driver**, **quick demo**, and a **bundled sample run** with figures for instant preview.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python scripts/run_quick_demo.py
```

Open the newest `runs/<timestamp>/` to see figures.

For a full interactive flow, open `notebooks/00_driver.ipynb` and run cells top‑to‑bottom (first cell auto‑installs the package for the notebook kernel).

A **sample run** with pre‑rendered figures lives in `runs/sample_run/` for immediate visual inspection.

![Figure: Spectral braid showing temporal evolution of eigenvalues.
Each colored line traces an eigenvalue \lambda_k as a function of time, illustrating how spectral modes evolve and occasionally exchange order. The gentle slopes indicate smooth eigenvalue dynamics, while the shaded regions mark intervals of potential mode interaction or near-degeneracy where eigenvalue trajectories approach or cross. This visualization highlights the continuity and topological braiding behavior of the spectrum over the sample run.](image.png)

![Figure: Continuity waterfall plot showing the temporal smoothness of mode evolution.
Each colored line represents the overlap magnitude $|\langle \varphi_k(t), \varphi_k(t+\Delta t) \rangle|$ for mode index $k = 1 \ldots 16$ across sequential frames. Values near 1 indicate high temporal continuity (smooth evolution), while deviations reflect transient changes or mixing between modes. The plot illustrates that most modes maintain strong continuity across time steps, suggesting stable mode tracking throughout the sample run.](image-1.png)