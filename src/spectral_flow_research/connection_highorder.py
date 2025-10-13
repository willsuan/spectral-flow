
import numpy as np
from .transport import polar_factor

def _log_orthogonal(M: np.ndarray) -> np.ndarray:
    """
    Matrix log for (near-)orthogonal M via eigen-decomposition.
    For unit-modulus eigenvalues w = e^{iθ}, log(w) = i θ with θ = angle(w) ∈ (-π, π].
    """
    w, V = np.linalg.eig(M.astype(complex))
    theta = np.angle(w)
    L = np.diag(1j*theta)  # log(e^{iθ}) = i θ
    Vinv = np.linalg.inv(V)
    A = V @ L @ Vinv
    return np.real_if_close(A, tol=1e-6)

def log_connection_from_S(S: np.ndarray, dt: float, mode: str = "polar") -> np.ndarray:
    """
    Higher-order connection:
      - mode='polar': A = log(polar(S)) / dt ∈ so(K)
      - mode='raw'  : A = log(S) / dt (less stable if S not near-orthogonal)
    Skew part is returned to suppress numerical drift.
    """
    M = polar_factor(S) if mode == "polar" else S
    try:
        from scipy.linalg import logm
        A = logm(M) / dt
    except Exception:
        A = _log_orthogonal(M) / dt
    A = np.real_if_close(A, tol=1e-8)
    return 0.5*(A - A.T)

def symmetric_curvature(A_series, dt: float):
    """
    Symmetric time discretization:
      F_i ≈ (A_{i+1} - A_{i-1})/(2dt) + A_i^2
    Returns array shape (T-2, K, K).
    """
    A = np.asarray(A_series)
    if A.shape[0] < 3: return np.zeros((0, A.shape[1], A.shape[2]))
    out = []
    for i in range(1, A.shape[0]-1):
        dA = (A[i+1] - A[i-1]) / (2.0*dt)
        out.append(dA + A[i] @ A[i])
    return np.array(out)
