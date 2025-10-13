
import numpy as np

def classical_mds(D: np.ndarray, dim: int = 2):
    """Classical MDS on distance matrix D."""
    K = D.shape[0]; J = np.eye(K) - np.ones((K,K))/K; B = -0.5 * J @ (D**2) @ J
    vals, vecs = np.linalg.eigh(B); order = np.argsort(vals)[::-1]; vals = vals[order]; vecs = vecs[:, order]; vals[vals<0]=0.0
    L = np.diag(np.sqrt(vals[:dim])); X = vecs[:, :dim] @ L; return X

def procrustes_align(A: np.ndarray, B: np.ndarray):
    """Orthogonal Procrustes aligning B to A."""
    M = A.T @ B; U,_,Vt = np.linalg.svd(M, full_matrices=False); R = U @ Vt; return B @ R

def mds_from_connection(A_hat: np.ndarray):
    """Embedding of modes using |A_hat| couplings as similarity (converted to a pseudodistance)."""
    W = np.abs(A_hat.copy()); np.fill_diagonal(W, 0.0); Wmax = np.max(W)+1e-12; D = np.sqrt(np.maximum(0.0, Wmax - W)); X = classical_mds(D, dim=2); return X
