
import numpy as np

def mass_energy(u: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum((u**2) * w))

def streaming_energy_defect(times, A_list, Phi_list, lam_list, w_list, u0, dt):
    """
    Compare energy for evolving basis vs implicit Euler step on A.
    """
    import numpy as np
    E_evo = np.zeros_like(times); E_euler = np.zeros_like(times)
    u_evo = u0.copy(); u_euler = u0.copy(); E_evo[0] = mass_energy(u_evo, w_list[0]); E_euler[0] = mass_energy(u_euler, w_list[0])
    for i in range(1, len(times)):
        A_prev = A_list[i-1]; Phi_prev, lam_prev = Phi_list[i-1], lam_list[i-1]; Phi_cur = Phi_list[i]
        c_prev = Phi_prev.T @ u_evo; c_cur = c_prev * np.exp(-lam_prev * dt); u_evo = Phi_cur @ c_cur
        try:
            import scipy.sparse as sp, scipy.sparse.linalg as spla
            N = u_euler.shape[0]; I = sp.identity(N, format="csr"); u_euler = spla.spsolve(I + dt*A_prev, u_euler)
        except Exception:
            # dense fallback
            A_dense = A_prev.toarray() if hasattr(A_prev, "toarray") else A_prev
            u_euler = np.linalg.solve(np.eye(A_dense.shape[0]) + dt*A_dense, u_euler)
        E_evo[i] = mass_energy(u_evo, w_list[i]); E_euler[i] = mass_energy(u_euler, w_list[i])
    return E_evo, E_euler, E_evo - E_euler
