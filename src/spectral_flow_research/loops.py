
from typing import List, Tuple
import numpy as np
from .shapes import polar_grid, radius_ellipse, radius_cardioid, radius_superellipse, radius_star, area_from_radius

LOOPS = {"quality_circle": ["ellipse","cardioid","ellipse"],
         "quality_star": ["star","superellipse","ellipse","star"],
         "ultra_tour": ["ellipse","cardioid","star","superellipse","ellipse"]}

def which_loop(name: str) -> List[str]:
    if name not in LOOPS: raise ValueError(f"Unknown loop '{name}'. Options: {list(LOOPS.keys())}")
    return LOOPS[name]

def segment_for_tau(path: List[str], tau: float) -> Tuple[int, float]:
    S = len(path) - 1; tau = np.clip(tau, 0.0, 1.0-1e-12); s = int(np.floor(tau * S)); alpha = (tau*S) - s; return s, alpha

def morph_masks_loop(n: int, loop_name: str, tau: float):
    path = which_loop(loop_name); s, alpha = segment_for_tau(path, tau); a, b = path[s], path[s+1]
    from .shapes import mask_from_radius
    X, Y, R, Theta, h = polar_grid(n)
    r_map = {
        "ellipse": lambda Th: radius_ellipse(Th, a=0.8, b=0.6),
        "cardioid": lambda Th: radius_cardioid(Th, a=0.6),
        "superellipse": lambda Th: radius_superellipse(Th, a=0.85, b=0.85, p=3.5),
        "star": lambda Th: radius_star(Th, base=0.75, amp=0.18, k=5),
    }
    Ra = r_map[a](Theta); Rb = r_map[b](Theta); R_tau = (1.0 - alpha)*Ra + alpha*Rb
    mask = (R <= R_tau).astype(np.uint8)
    return mask, h, a, b, alpha

def morph_masks_loop_equal_area(n: int, loop_name: str, tau: float):
    """Equal-area scaling along the loop: rescale r to keep area constant (area from first vertex)."""
    path = which_loop(loop_name); s, alpha = segment_for_tau(path, tau); a, b = path[s], path[s+1]
    X, Y, R, Theta, h = polar_grid(n)
    r_map = {
        "ellipse": lambda Th: radius_ellipse(Th, a=0.8, b=0.6),
        "cardioid": lambda Th: radius_cardioid(Th, a=0.6),
        "superellipse": lambda Th: radius_superellipse(Th, a=0.85, b=0.85, p=3.5),
        "star": lambda Th: radius_star(Th, base=0.75, amp=0.18, k=5),
    }
    R0 = r_map[path[0]](Theta); A0 = area_from_radius(R0, Theta)
    Ra = r_map[a](Theta); Rb = r_map[b](Theta); R_tau = (1.0 - alpha)*Ra + alpha*Rb
    Atau = area_from_radius(R_tau, Theta) + 1e-16; gamma = np.sqrt(A0 / Atau)
    R_tau_scaled = gamma * R_tau
    mask = (R <= R_tau_scaled).astype(np.uint8)
    return mask, h, a, b, alpha, A0, Atau, gamma
