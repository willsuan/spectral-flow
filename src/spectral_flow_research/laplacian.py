
import numpy as np, scipy.sparse as sp
from typing import Tuple

def build_dirichlet_laplacian(mask: np.ndarray, h: float) -> Tuple[sp.csr_matrix, np.ndarray]:
    """5-point Laplacian on interior nodes of 'mask' (1=inside)."""
    n, m = mask.shape; inside = mask.astype(bool)
    interior = np.flatnonzero(inside.ravel())
    map_ = -np.ones(n*m, dtype=int); map_[interior] = np.arange(interior.size, dtype=int)
    rows, cols, data = [], [], []
    def lin(i,j): return i*m + j
    for i in range(n):
        for j in range(m):
            if not inside[i,j]: continue
            p = map_[lin(i,j)]; diag = 4.0
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ii, jj = i+di, j+dj
                if 0<=ii<n and 0<=jj<m and inside[ii,jj]:
                    q = map_[lin(ii,jj)]; rows.append(p); cols.append(q); data.append(-1.0)
            rows.append(p); cols.append(p); data.append(diag)
    A = sp.csr_matrix((np.array(data)/h**2, (rows, cols)), shape=(interior.size, interior.size))
    return A, interior
