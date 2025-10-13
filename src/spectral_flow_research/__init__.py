
"""
spectral_flow_research v1.0 — documented research kit for spectral flow on evolving domains.

Subpackages / Modules
---------------------
- shapes: analytic shapes, masks, area
- geometry: signed distance, central gradients
- laplacian: Dirichlet 5-point Laplacian on interior points
- eigsolve: eigensolver (SciPy if available, dense NumPy fallback), Gram–Schmidt, mode matching
- transport: gauges (polar / sign-match), polar factor
- connection: discrete connections (overlap), curvature; high-order (log-polar, symmetric curvature)
- hadamard: shape derivative (fixed/adaptive σ), curvature, kinematics with timing profiles
- holonomy: product, invariants; visuals
- mds: mode geometry embedding + Procrustes alignment
- energy: energy defect benchmark (evolving basis vs implicit Euler)
- ablation, convergence: scripts and helpers
- reproducibility: seeding, environment manifest; figure stamps are set at experiment entry
- math_checks: sanity checks and reports
- experiment: main CLI runner (writes arrays + figures to runs/<timestamp>/)

Each module features docstrings connecting code to the underlying mathematics.
"""
__version__ = "1.0.0"
