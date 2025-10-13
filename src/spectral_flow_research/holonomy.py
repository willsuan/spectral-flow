
import numpy as np
from typing import List, Tuple

def holonomy_product(S_list: List[np.ndarray]) -> np.ndarray:
    """P = ∏ S_i (right-to-left time order)."""
    P = np.eye(S_list[0].shape[0], dtype=complex)
    for S in S_list: P = P @ S
    return P

def permutation_from_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Permutation by row argmax of |P| plus weights."""
    V = np.abs(P); perm = np.argmax(V, axis=1); weights = V[np.arange(V.shape[0]), perm]; return perm, weights

def holonomy_invariants(P: np.ndarray):
    """Return diag(P), eigenvalues, row-argmax permutation + weights."""
    diag = np.diag(P); evals = np.linalg.eigvals(P); perm, w = permutation_from_matrix(P); return diag, evals, perm, w
