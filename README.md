# Visualizing Spectral Flow and Harmonic Transport

This is an ongiong project to numerically analyze and visualize the spectral flow of the Dirichlet Laplacian
$-\Delta_{\Omega(t)}$ on a smoothly evolving family of planar domains $\{\Omega(t)\}_{t\in[0,T]}$.
We develop numerical and geometric tools to study how eigenvalues and eigenfunctions evolve
as the shape changes, defining a connection and curvature on the bundle of eigenspaces.

This provides a computational differential geometry of eigenmodes — a way to quantify
how modes rotate, permute, and accumulate geometric phase under deformations of the domain.

Functionality:
- Core pipeline: evolving masks → Laplacians → eigen-tracking → connections/curvature → holonomy → rich figures.
- Gauge choices: overlap vs log‑polar (Lie‑algebra) connection with symmetric curvature.
- Hadamard shape derivative (fixed/adaptive δ‑band) + timing profiles for boundary velocity/acceleration.
- Equal‑area loops, shape‑space metrics & geodesic paths.
- Ablations, convergence probes, repro manifests, figure stamps.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
python scripts/run_quick_demo.py
```

Open the newest `runs/<timestamp>/` to see figures.

For a full interactive flow, open `notebooks/00_driver.ipynb` and run cells top‑to‑bottom (first cell auto‑installs the package for the notebook kernel).

---

Below we discuss some of the mathematical theory.

### Laplacian Eigenproblem on an Evolving Domain

For each $t$, the Dirichlet eigenproblem is

$$
-\Delta \phi_k(t,x) = \lambda_k(t)\,\phi_k(t,x), \quad
\phi_k|_{\partial\Omega(t)} = 0, \quad
\|\phi_k(t)\|_{L^2(\Omega(t))}=1.
$$

The functions $\{\phi_k(t)\}$ form an orthonormal basis of $L^2(\Omega(t))$.
Tracking their evolution defines the spectral flow $\lambda_k(t)$ and mode transport.

---

### Hadamard Shape Derivative

If the boundary moves with normal velocity $V_n = V\!\cdot\!n$,
the Hadamard formula gives the eigenvalue variation:

$$
\frac{d\lambda_k}{dt}
  = -\!\!\int_{\partial\Omega(t)} (\partial_n\phi_k)^2\, V_n\, ds.
$$

Numerically, this is approximated with a narrow Gaussian band around the boundary:

$$
\int_{\partial\Omega} f\,ds
  \approx \int_{\mathbb{R}^2} f(x)\,\delta_\sigma(s(x))\,|\nabla s(x)|\,dx, \quad
\delta_\sigma(s)=\frac{1}{\sqrt{2\pi}\sigma}e^{-s^2/(2\sigma^2)}.
$$

An adaptive bandwidth $\sigma(\theta)=\sigma_0 h/(1+\alpha|\kappa(\theta)|)$
reduces bias near high-curvature boundary regions.

---

### Spectral Connection and Curvature

Let $\Phi(t)=[\phi_1(t),\dots,\phi_K(t)]$ and $M$ the discrete mass matrix.
Define the overlap

$$
S(t,t+\Delta t) = \Phi(t)^\top M \Phi(t+\Delta t).
$$

For small $\Delta t$, we expand $S = I + \Delta t\,\widehat{\mathcal A} + O(\Delta t^2)$,
so that

$$
\widehat{\mathcal A}(t) \approx \frac{S - I}{\Delta t}
$$

is a discrete connection describing infinitesimal rotation of the basis.

#### Lie–algebra (log-polar) connection

To ensure gauge-consistency, we define

$$
\widehat{\mathcal A}_{\log}
   = \frac{1}{\Delta t}\log(\mathrm{polar}(S)) \in \mathfrak{so}(K),
$$

the logarithm of the orthogonal polar factor of $S$.  
This lives naturally in the Lie algebra of skew-symmetric matrices.

#### Curvature

Discrete curvature measures non-commutativity of parallel transport:

$$
\widehat{\mathcal F}_i
  \approx \frac{A_{i+1}-A_{i-1}}{2\,\Delta t} + A_i^2,
$$

analogous to the continuous identity $\mathcal F = \dot{\mathcal A} + \mathcal A^2$.

---

### Holonomy

Along a closed deformation loop, the overlaps compose into the holonomy

$$
P = \prod_i S_i,
$$

a discrete path-ordered exponential.  
The unitary part of $P$ encodes mode permutation and geometric phase accumulation.

- $|P|$ ≈ permutation matrix ⇒ mode relabeling after a cycle.  
- $\mathrm{diag}(P)$ phases ⇒ Berry-type geometric phases.  
- $\|P^\top P - I\|$ measures orthogonality loss (numerical error).

---

### Energy Flow and Evolving Basis

For heat evolution $u_t = \Delta_{\Omega(t)}u$, project onto the instantaneous basis:
$$\dot c_k = -\lambda_k c_k.$$
Integrating gives
$$c_k(t+\Delta t)=c_k(t)e^{-\lambda_k(t)\Delta t}.$$

Comparing this evolving-basis propagation to an implicit Euler time step on the full grid
defines the energy defect — a measure of how well the evolving basis tracks the true dynamics.

---

### Equal-Area Loops and Shape-Space Geometry

To remove trivial scaling effects, the interpolated radius $r(\theta,t)$
is rescaled to keep the area constant:

$$
A(t)=\tfrac12\int_0^{2\pi}r(\theta,t)^2\,d\theta,\quad
r\mapsto \gamma(t)r,\quad \gamma(t)=\sqrt{\frac{A(0)}{A(t)}}.
$$

Shape families (ellipses, cardioids, stars) form a shape manifold
with $L^2$ metric

$$
\langle \delta r_1,\delta r_2\rangle = \frac{1}{2\pi}\int_0^{2\pi}\delta r_1(\theta)\delta r_2(\theta)\,d\theta,
$$

yielding geodesics $r_s=(1-s)r_0+s r_1$ and local metric tensors
$G_{ij}=\langle \partial r/\partial p_i,\partial r/\partial p_j\rangle.$

---

### Computational Geometry Summary

| Concept | Continuous Object | Discrete Analogue |
|----------|-------------------|-------------------|
| Eigenfunction basis | $\{\phi_k(t)\}\subset L^2(\Omega(t))$ | Grid vectors $\Phi(t)$ |
| Inner product | $\langle f,g\rangle = \int_\Omega fg$ | Weighted dot product $f^\top M g$ |
| Connection | $\mathcal A_t = \Phi^\top \dot\Phi$ | $A = (S-I)/\Delta t$ or $\log(\mathrm{polar}(S))/\Delta t$ |
| Curvature | $\mathcal F = \dot{\mathcal A}+\mathcal A^2$ | Finite difference stencil |
| Holonomy | $\mathcal{P}\exp\!\int\mathcal{A}\,dt$ | Product $P=\prod S_i$ |
| Shape derivative | Hadamard integral | Boundary narrow-band quadrature |

**Spectral braid**.
Time evolution of the ordered Dirichlet–Laplacian eigenvalues $\lambda_k(t)$ for a smoothly deforming planar domain.  Each colored curve traces one eigenvalue branch, illustrating avoided crossings and level repulsion as the geometry evolves.  Shaded vertical bands mark timesteps where eigenvalue orderings interchange—“braiding’’ events that encode the spectral holonomy of the deformation.
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/bb7e8fb8-167d-4fdf-9deb-035162b14bf9" />

**Continuity waterfall** $|\langle\varphi_k(t),\varphi_k(t+\Delta t)\rangle|$.
Mode-overlap continuity between successive frames for the first 16 eigenfunctions. Each horizontal trace (indexed by k) shows the self-overlap magnitude of mode k across time, quantifying how smoothly eigenfunctions are transported.  Overlap values near 1 indicate adiabatic tracking; small dips correspond to near-degeneracies or mode mixing identified in the spectral-braid plot.
<img width="1200" height="1110" alt="image-1" src="https://github.com/user-attachments/assets/9d0bc1b2-32b2-4558-8771-4c94df0101b7" />


## References

1. J. Hadamard, *Mémoire sur le problème d’analyse relatif à l’équilibre des plaques élastiques encastrées* (1908).  
2. M. V. Berry, *Quantal phase factors accompanying adiabatic changes*, Proc. R. Soc. Lond. A 392 (1984).
3. 	S. C. Eisenstat & H. C. Elman, “Inexact Krylov Subspace Methods for Nonsymmetric Linear Systems,” SIAM Journal on Matrix Analysis and Applications, 21(4), 2000.
4. 	M. E. Hochstenbach & F. J. C. Kraaijevanger, “A Continuation Method for Eigenpairs of Parametrized Hermitian Matrices,” Linear Algebra and its Applications, 432 (2010), 1425–1445.
5. 	G. W. Stewart, Perturbation Theory for the Singular Value Decomposition, in Numerical Linear Algebra, SIAM (1999).
6. J. B. Keller & S. I. Rubinow, *Asymptotic solution of eigenvalue problems*, Ann. Phys. 9 (1960).

---
