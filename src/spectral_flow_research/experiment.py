
import os, json, numpy as np, datetime as _dt, math
from typing import Optional
from tqdm import tqdm

from .shapes import morph_masks
from .laplacian import build_dirichlet_laplacian
from .eigsolve import eigsh_auto, gram_schmidt, match_modes, transport_basis, weighted_inner
from .transport import apply_polar_gauge, sign_match_gauge
from .connection import connection_overlap, curvature_midpoint
from .connection_highorder import log_connection_from_S, symmetric_curvature
from .hadamard import boundary_normal_velocity_field, hadamard_shape_derivative, hadamard_shape_derivative_adaptive, boundary_normal_kinematics
from .holonomy import holonomy_product, holonomy_invariants
from .viz import spectral_braid, continuity_waterfall, chord_from_matrix, holonomy_phase_wheel, holonomy_perm_heat
from .reproducibility import init_seed, write_manifest, stamp_from_manifest

def embed_to_grid(vec: np.ndarray, interior_idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n*n,), dtype=float); out[interior_idx] = vec; return out.reshape(n, n)

def make_run_dir(root: str = "runs") -> str:
    os.makedirs(root, exist_ok=True); ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, ts); os.makedirs(path, exist_ok=True); return path

def run(n: int = 128, K: int = 16, frames: int = 36, T: float = 0.6,
        shape_from: str = "ellipse", shape_to: str = "cardioid",
        gauge: str = "polar", connection_mode: str = "log_polar",
        hadamard: str = "adaptive", morph_timing: str = "cubic",
        seed: int = 2025, outdir: Optional[str] = None, equal_area: bool = False) -> str:
    """
    Full experiment driver.
    - eigen-tracking with seeded v0 for stability
    - gauge: 'polar' (recommended) or 'match'
    - connection_mode: 'overlap' or 'log_polar'/'log_raw'
    - hadamard: 'off'|'fixed'|'adaptive' (computes λ' Hadamard predictor)
    - morph_timing: 'linear'|'cubic' (for Vn/An fields when hadamard != 'off')
    - equal_area: if True, rescales r(θ) to keep area constant for two-shape morphs
    """
    rng = init_seed(seed); run_dir = outdir or make_run_dir("runs")
    times = np.linspace(0.0, T, frames); dt = times[1]-times[0] if frames>1 else T
    lambdas = np.zeros((frames, K)); cont = np.zeros((frames-1, K))
    A_hat_series = []; F_series = []; S_list = []

    # Repro manifest + stamp
    manifest = write_manifest(run_dir, seed=seed, extra=dict(version="1.0.0"))
    stamp = stamp_from_manifest(manifest, run_dir)

    Phi_prev = None; w_prev = None; interior_prev = None
    for it, t in enumerate(tqdm(times, desc="frames")):
        tau = t / max(T, 1e-12)
        if equal_area:
            # quick equal-area by rescaling morph radius (two-shape case)
            from .shapes import polar_grid, radius_ellipse, radius_cardioid, radius_superellipse, radius_star, area_from_radius, mask_from_radius
            X, Y, R, Theta, h = polar_grid(n)
            r_map = {
                "ellipse": lambda Th: radius_ellipse(Th, a=0.8, b=0.6),
                "cardioid": lambda Th: radius_cardioid(Th, a=0.6),
                "superellipse": lambda Th: radius_superellipse(Th, a=0.85, b=0.85, p=3.5),
                "star": lambda Th: radius_star(Th, base=0.75, amp=0.18, k=5),
            }
            R0 = r_map[shape_from](Theta); A0 = area_from_radius(R0, Theta)
            R1 = r_map[shape_to](Theta); R_tau = (1.0 - tau)*R0 + tau*R1
            Atau = area_from_radius(R_tau, Theta) + 1e-16; gamma = np.sqrt(A0/Atau); R_tau *= gamma
            mask = mask_from_radius(R, R_tau)
        else:
            mask, h = morph_masks(n, shape_from, shape_to, tau)
        A, interior = build_dirichlet_laplacian(mask, h)
        v0 = rng.standard_normal(A.shape[0]); lam, Phi = eigsh_auto(A, K=K, which="SA", v0=v0)
        w = np.ones(Phi.shape[0])*(h**2); Phi = gram_schmidt(Phi, w)
        lambdas[it,:] = lam

        if it>0:
            perm, sgn = match_modes(Phi_prev, Phi, w_prev); Phi_aligned = transport_basis(Phi, perm, sgn)
            S_raw = weighted_inner(Phi_prev, Phi_aligned, w_prev)
            if gauge == "polar": Phi_aligned = apply_polar_gauge(Phi_aligned, S_raw)
            else:                Phi_aligned = sign_match_gauge(Phi_prev, Phi_aligned, w_prev)
            S = weighted_inner(Phi_prev, Phi_aligned, w_prev); S_list.append(S)
            cont[it-1,:] = np.abs(np.diag(S))
            # connection
            if connection_mode == "overlap": A_hat = connection_overlap(S, dt)
            elif connection_mode == "log_polar": A_hat = log_connection_from_S(S, dt, mode="polar")
            else: A_hat = log_connection_from_S(S, dt, mode="raw")
            A_hat_series.append(A_hat)
            if len(A_hat_series)>=2:
                F_series.append(curvature_midpoint(A_hat_series[-2], A_hat_series[-1], dt))
        Phi_prev, w_prev, interior_prev = Phi, w, interior

    # Save arrays
    np.save(os.path.join(run_dir, "lambdas.npy"), lambdas); np.save(os.path.join(run_dir, "continuity.npy"), cont)
    if A_hat_series:
        np.save(os.path.join(run_dir, "connection_Ahat.npy"), np.stack(A_hat_series, axis=0))
        if len(F_series)>0: np.save(os.path.join(run_dir, "connection_curvature.npy"), np.stack(F_series, axis=0))
        # symmetric curvature (Lie) for log connection
        if connection_mode in ("log_polar","log_raw") and len(A_hat_series)>2:
            Fsym = symmetric_curvature(A_hat_series, dt)
            if Fsym.shape[0]>0: np.save(os.path.join(run_dir, "connection_curvature_sym.npy"), Fsym)

    # Figures
    spectral_braid(times, lambdas, stamp_text=stamp, out=os.path.join(run_dir, "spectral_braid.png"))
    continuity_waterfall(cont, stamp_text=stamp, out=os.path.join(run_dir, "continuity_waterfall.png"))
    if A_hat_series:
        chord_from_matrix(A_hat_series[-1], "Chord(|Â| last)", stamp_text=stamp, out=os.path.join(run_dir, "connection_chord_Ahat_last.png"))

    # Holonomy
    if len(S_list)>0:
        P = holonomy_product(S_list); np.save(os.path.join(run_dir, "holonomy_P.npy"), P)
        dP, ev, perm, wts = holonomy_invariants(P)
        holonomy_phase_wheel(dP, stamp_text=stamp, out=os.path.join(run_dir, "holonomy_phase_wheel.png"))
        holonomy_perm_heat(P, stamp_text=stamp, out=os.path.join(run_dir, "holonomy_permutation.png"))

    # Config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(n=n, K=K, frames=frames, T=T, shape_from=shape_from, shape_to=shape_to,
                       gauge=gauge, connection_mode=connection_mode, hadamard=hadamard,
                       morph_timing=morph_timing, equal_area=equal_area, seed=seed), f, indent=2)
    return run_dir

if __name__ == "__main__":
    # Minimal CLI run
    rd = run()
    print("[OK] Run saved at:", rd)
