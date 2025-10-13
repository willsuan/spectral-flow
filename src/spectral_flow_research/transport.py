
import numpy as np

def polar_factor(S: np.ndarray) -> np.ndarray:
    """Orthogonal polar factor of S via SVD: Q = U V^T."""
    U, _, Vt = np.linalg.svd(S, full_matrices=False); return U @ Vt

def apply_polar_gauge(Phi_tp: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Right-multiply Φ(t+Δt) by polar factor of overlap to minimize misalignment (discrete parallel transport)."""
    Q = polar_factor(S); return Phi_tp @ Q

def sign_match_gauge(Phi_t: np.ndarray, Phi_tp: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Flip signs to maximize diagonal overlaps only (baseline gauge)."""
    G = (Phi_t * w.reshape(-1,1)).T @ Phi_tp; sgn = np.sign(np.diag(G)); sgn[sgn==0]=1.0
    return Phi_tp * sgn.reshape(1,-1)
