
import numpy as np

def connection_overlap(S: np.ndarray, dt: float) -> np.ndarray:
    """First-order connection from overlaps: A ≈ (S - I)/dt."""
    K = S.shape[0]; return (S - np.eye(K)) / dt

def curvature_midpoint(A_prev: np.ndarray, A_next: np.ndarray, dt: float) -> np.ndarray:
    """Midpoint curvature: F ≈ (A_next - A_prev)/(2dt) + A_mid^2, A_mid=(A_prev+A_next)/2."""
    A_mid = 0.5*(A_prev + A_next); dA = (A_next - A_prev)/(2.0*dt); return dA + A_mid @ A_mid
