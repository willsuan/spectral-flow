
import numpy as np

def eigen_residuals(A, Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """||A φ_k - λ_k φ_k||_2 per mode (dense multiply if needed)."""
    if hasattr(A, "dot"): R = A.dot(Phi) - Phi * lam.reshape(1,-1)
    else: R = A @ (Phi) - Phi * lam.reshape(1,-1)
    return np.linalg.norm(R, axis=0)

def gram_error(Phi: np.ndarray, w: np.ndarray) -> float:
    G = (Phi * w.reshape(-1,1)).T @ Phi; return float(np.linalg.norm(np.eye(Phi.shape[1]) - G, ord="fro"))

def a_hat_skew_error(A_hat: np.ndarray) -> float:
    num = np.linalg.norm(A_hat + A_hat.T, ord="fro"); den = np.linalg.norm(A_hat, ord="fro") + 1e-16; return float(num/den)

def hadamard_fd_report(times: np.ndarray, lambdas: np.ndarray, hadamard_dlambda: np.ndarray):
    if hadamard_dlambda is None or lambdas.shape[0] < 3:  return {"available": False}
    dt = times[1]-times[0]; fd = (lambdas[2:] - lambdas[:-2])/(2*dt); H = hadamard_dlambda[1:-1, :]
    corrs, rmses = [], []
    for k in range(lambdas.shape[1]):
        x, y = fd[:,k], H[:,k]; xm, ym = x - x.mean(), y - y.mean(); den = (np.linalg.norm(xm)*np.linalg.norm(ym) + 1e-16)
        corr = float((xm @ ym) / den) if den>0 else 0.0; corrs.append(corr); rmses.append(float(np.sqrt(np.mean((x - y)**2))))
    return {"available": True, "corr_mean": float(np.mean(corrs)), "corr_min": float(np.min(corrs)), "rms_mean": float(np.mean(rmses))}

def holonomy_quality(P: np.ndarray):
    I = np.eye(P.shape[0]); ortho_err = float(np.linalg.norm(P.T.conj() @ P - I, ord="fro") / (np.linalg.norm(I, ord="fro")+1e-16))
    evals = np.linalg.eigvals(P); unit_mod_dev = float(np.mean(np.abs(np.abs(evals) - 1.0))); return {"ortho_err": ortho_err, "unit_modulus_mean_dev": unit_mod_dev}
