
import numpy as np

def eigsh_auto(A, K: int, which: str = "SA", v0=None):
    """Use SciPy's eigsh when available; otherwise dense eigh fallback (for small demos)."""
    try:
        import scipy.sparse.linalg as spla
        vals, vecs = spla.eigsh(A, k=K, which=which, v0=v0)
        idx = np.argsort(vals)
        return np.real(vals[idx]), np.real(vecs[:, idx])
    except Exception:
        A_dense = A.toarray() if hasattr(A, "toarray") else np.array(A)
        vals, vecs = np.linalg.eigh(A_dense)
        idx = np.argsort(vals); return np.real(vals[idx[:K]]), np.real(vecs[:, idx[:K]])

def gram_schmidt(Phi: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Weighted Gram–Schmidt w.r.t. mass weights w (uniform grid ⇒ w=h^2)."""
    Q = np.array(Phi, copy=True); W = w.reshape(-1,1)
    for k in range(Q.shape[1]):
        for j in range(k):
            r = np.sum(W[:,0] * Q[:,j] * Q[:,k]); Q[:,k] -= r * Q[:,j]
        nrm = np.sqrt(np.sum(W[:,0] * Q[:,k]**2))
        if nrm < eps: raise RuntimeError("Gram-Schmidt breakdown")
        Q[:,k] /= nrm
    return Q

def weighted_inner(Phi: np.ndarray, Psi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(Φ^T M Ψ) with diagonal mass weights M=diag(w)."""
    return (Phi * w.reshape(-1,1)).T @ Psi

def match_modes(Phi_t: np.ndarray, Phi_tp: np.ndarray, w: np.ndarray):
    """Match modes by maximizing |〈φ_k(t), φ_j(t+Δt)〉|. Uses Hungarian if SciPy available; greedy fallback otherwise."""
    G = weighted_inner(Phi_t, Phi_tp, w)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-np.abs(G))
        sgn = np.sign(G[row_ind, col_ind]); sgn[sgn==0]=1.0
        return col_ind, sgn
    except Exception:
        # greedy fallback
        K = G.shape[0]; used = set(); perm = np.zeros(K, dtype=int); sgn = np.ones(K)
        for k in range(K):
            j = int(np.argmax(np.abs(G[k,:])))
            while j in used:
                G[k,j] = 0.0; j = int(np.argmax(np.abs(G[k,:])))
            used.add(j); perm[k] = j; sgn[k] = np.sign(G[k,j]) if G[k,j]!=0 else 1.0
        return perm, sgn

def transport_basis(Phi_tp: np.ndarray, perm, sgn):
    """Apply permutation and signs to align the basis at t+Δt to t."""
    return Phi_tp[:, perm] * sgn.reshape(1,-1)
