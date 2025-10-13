
import numpy as np
from typing import Tuple, Literal
ShapeName = Literal["ellipse", "cardioid", "superellipse", "star"]

def polar_grid(n: int):
    """Uniform grid on [-1,1]^2 with polar coords. Returns X,Y,R,Theta,h."""
    x = np.linspace(-1.0, 1.0, n); y = np.linspace(-1.0, 1.0, n)
    h = x[1] - x[0]; X,Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2); Theta = np.arctan2(Y, X); return X, Y, R, Theta, h

def radius_ellipse(theta: np.ndarray, a: float = 0.8, b: float = 0.6) -> np.ndarray:
    """Radius of ellipse boundary in polar form."""
    ct, st = np.cos(theta), np.sin(theta)
    denom = (ct**2)/(a**2) + (st**2)/(b**2)
    return 1.0 / np.sqrt(denom + 1e-16)

def radius_cardioid(theta: np.ndarray, a: float = 0.6) -> np.ndarray:
    """Cardioid radius: r(θ) = a (1 - cos θ)."""
    return a * (1.0 - np.cos(theta))

def radius_superellipse(theta: np.ndarray, a: float = 0.85, b: float = 0.85, p: float = 3.5) -> np.ndarray:
    """Superellipse radius by L^p norm."""
    ct, st = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    return 1.0 / ((ct**p)/(a**p) + (st**p)/(b**p))**(1.0/p)

def radius_star(theta: np.ndarray, base: float = 0.75, amp: float = 0.18, k: int = 5) -> np.ndarray:
    """Star perturbation of a disk."""
    return base * (1.0 + amp * np.cos(k * theta))

def mask_from_radius(R: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Binary mask for interior of r(θ)."""
    return (R <= Rb).astype(np.uint8)

def make_mask(n: int, shape: ShapeName, **kwargs) -> Tuple[np.ndarray, float]:
    """Return (mask, h) for a named analytic shape with default parameters."""
    X, Y, R, Theta, h = polar_grid(n)
    if shape == "ellipse": Rb = radius_ellipse(Theta, **{k:v for k,v in kwargs.items() if k in ("a","b")})
    elif shape == "cardioid": Rb = radius_cardioid(Theta, **{k:v for k,v in kwargs.items() if k in ("a",)})
    elif shape == "superellipse": Rb = radius_superellipse(Theta, **{k:v for k,v in kwargs.items() if k in ("a","b","p")})
    elif shape == "star": Rb = radius_star(Theta, **{k:v for k,v in kwargs.items() if k in ("base","amp","k")})
    else: raise ValueError(f"Unknown shape {shape}")
    return mask_from_radius(R, Rb), h

def morph_masks(n: int, shape_from: ShapeName, shape_to: ShapeName, t: float) -> Tuple[np.ndarray, float]:
    """Linear interpolation in r(θ) between two analytic shapes (family defaults)."""
    X, Y, R, Theta, h = polar_grid(n)
    R1 = {"ellipse": radius_ellipse(Theta, a=0.8, b=0.6),
          "cardioid": radius_cardioid(Theta, a=0.6),
          "superellipse": radius_superellipse(Theta, a=0.85, b=0.85, p=3.5),
          "star": radius_star(Theta, base=0.75, amp=0.18, k=5)}[shape_from]
    R2 = {"ellipse": radius_ellipse(Theta, a=0.8, b=0.6),
          "cardioid": radius_cardioid(Theta, a=0.6),
          "superellipse": radius_superellipse(Theta, a=0.85, b=0.85, p=3.5),
          "star": radius_star(Theta, base=0.75, amp=0.18, k=5)}[shape_to]
    Rb = (1.0 - t)*R1 + t*R2
    return mask_from_radius(R, Rb), h

def area_from_radius(r: np.ndarray, theta: np.ndarray) -> float:
    """Polar area: A = 1/2 ∫ r(θ)^2 dθ (integrate along θ axis)."""
    if r.ndim == 2:
        th = theta[0, :]; rsq = (r**2).mean(axis=0)
        return 0.5 * float(np.trapz(rsq, th))
    return 0.5 * float(np.trapz(r**2, theta))
