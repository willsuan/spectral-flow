
import numpy as np
from typing import Literal
from .shapes import polar_grid, radius_ellipse, radius_cardioid, radius_superellipse, radius_star
from .geometry import signed_distance, gradient_central, delta_kernel

ShapeName = Literal["ellipse", "cardioid", "superellipse", "star"]

def _radius(shape: ShapeName, theta: np.ndarray) -> np.ndarray:
    if shape == "ellipse": return radius_ellipse(theta, a=0.8, b=0.6)
    if shape == "cardioid": return radius_cardioid(theta, a=0.6)
    if shape == "superellipse": return radius_superellipse(theta, a=0.85, b=0.85, p=3.5)
    if shape == "star": return radius_star(theta, base=0.75, amp=0.18, k=5)
    raise ValueError(f"Unknown shape: {shape}")

def _dR_dtheta(shape: ShapeName, theta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    rp = _radius(shape, theta + eps); rm = _radius(shape, theta - eps); return (rp - rm)/(2.0*eps)

def _d2R_dtheta2(shape: ShapeName, theta: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    rp = _dR_dtheta(shape, theta + eps); rm = _dR_dtheta(shape, theta - eps); return (rp - rm)/(2.0*eps)

def curvature_polar(shape: ShapeName, theta: np.ndarray) -> np.ndarray:
    """κ(θ) = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}."""
    r = _radius(shape, theta); rp = _dR_dtheta(shape, theta); rpp = _d2R_dtheta2(shape, theta)
    denom = (r**2 + rp**2)**1.5 + 1e-12; num = r**2 + 2*(rp**2) - r*rpp; return num / denom

def delta_kernel_variable(s: np.ndarray, sigma_field: np.ndarray):
    return (1.0/(np.sqrt(2.0*np.pi)*sigma_field)) * np.exp(-0.5*(s/sigma_field)**2)

def hadamard_shape_derivative(phi_grid: np.ndarray, mask: np.ndarray, h: float, Vn: np.ndarray,
                              sigma_factor: float = 1.5) -> float:
    """Hadamard first variation: ∫_{∂Ω} (∂_n φ)^2 V_n δ_σ(s) dA, approximated on grid."""
    s = signed_distance(mask, h); sigma = sigma_factor * h
    sx, sy = gradient_central(s, h); nrm = np.sqrt(sx**2 + sy**2) + 1e-12; nx, ny = sx/nrm, sy/nrm
    phix, phiy = gradient_central(phi_grid, h); dn_phi = phix*nx + phiy*ny; delta = delta_kernel(s, sigma)
    integrand = (dn_phi**2) * Vn * delta * nrm; return float(- np.sum(integrand) * (h**2))

def hadamard_shape_derivative_adaptive(phi_grid: np.ndarray, mask: np.ndarray, h: float, Vn: np.ndarray,
                                       shape_from: ShapeName, shape_to: ShapeName, tau: float,
                                       sigma0: float = 1.5, alpha: float = 0.5, clip_bounds=(0.75, 2.5)) -> float:
    """Adaptive σ(θ) shrinks in high curvature regions to reduce bias."""
    X, Y, Rgrid, Theta, _ = polar_grid(mask.shape[0])
    kappa_from = curvature_polar(shape_from, Theta); kappa_to = curvature_polar(shape_to, Theta); kappa = (1.0 - tau)*kappa_from + tau*kappa_to
    sigma_field = sigma0*h / (1.0 + alpha*np.abs(kappa)); sigma_field = np.clip(sigma_field, clip_bounds[0]*h, clip_bounds[1]*h)
    s = signed_distance(mask, h); sx, sy = gradient_central(s, h); nrm = np.sqrt(sx**2 + sy**2) + 1e-12; nx, ny = sx/nrm, sy/nrm
    phix, phiy = gradient_central(phi_grid, h); dn_phi = phix*nx + phiy*ny; delta = delta_kernel_variable(s, sigma_field)
    integrand = (dn_phi**2) * Vn * delta * nrm; return float(- np.sum(integrand) * (h**2))

def timing_profile(tau: float, mode: str = "linear"):
    """Return s, s', s'' for timing profiles: linear or cubic (C1 ease in/out)."""
    if mode == "linear": return tau, 1.0, 0.0
    if mode in ("cubic", "cubic_smooth"):
        s = 3*tau*tau - 2*tau*tau*tau; sd = 6*tau - 6*tau*tau; sdd = 6 - 12*tau; return s, sd, sdd
    return tau, 1.0, 0.0

def boundary_normal_velocity_field(n: int, shape_from: ShapeName, shape_to: ShapeName, tau: float, T_phys: float):
    """Legacy (linear timing) velocity field on boundary."""
    X, Y, Rgrid, Theta, h = polar_grid(n); R_from = _radius(shape_from, Theta); R_to = _radius(shape_to, Theta)
    R_tau = (1.0 - tau) * R_from + tau * R_to; dR_dt = (R_to - R_from) / max(T_phys, 1e-12)
    dR_dth = _dR_dtheta(shape_from, Theta)*(1.0 - tau) + _dR_dtheta(shape_to, Theta)*tau
    denom = np.sqrt(R_tau**2 + dR_dth**2) + 1e-12; Vn = dR_dt * (R_tau / denom); return Vn, h

def boundary_normal_kinematics(n: int, shape_from: ShapeName, shape_to: ShapeName, tau: float, T_phys: float, timing: str = "linear"):
    """Velocity & acceleration under timing profile; useful for 2nd-order λ checks."""
    X, Y, Rgrid, Theta, h = polar_grid(n); R_from = _radius(shape_from, Theta); R_to = _radius(shape_to, Theta)
    s, sd, sdd = timing_profile(tau, mode=timing)
    R_tau = (1.0 - s) * R_from + s * R_to
    dR_dt = (R_to - R_from) * (sd / max(T_phys, 1e-12))
    d2R_dt2 = (R_to - R_from) * (sdd / max(T_phys**2, 1e-12))
    dR_dth = _dR_dtheta(shape_from, Theta)*(1.0 - s) + _dR_dtheta(shape_to, Theta)*s
    denom = (np.sqrt(R_tau**2 + dR_dth**2) + 1e-12)
    Vn = dR_dt * (R_tau / denom); An = d2R_dt2 * (R_tau / denom); return Vn, An, h
