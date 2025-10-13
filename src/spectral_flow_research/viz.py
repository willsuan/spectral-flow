
import numpy as np, matplotlib.pyplot as plt
from .viz_utils import stamp

def spectral_braid(times: np.ndarray, lambdas: np.ndarray, stamp_text=None, out=None):
    fig, ax = plt.subplots(figsize=(8,4))
    for k in range(lambdas.shape[1]): ax.plot(times, lambdas[:,k], lw=1.2)
    gaps = lambdas[:,1:] - lambdas[:,:-1]; gmin = np.min(gaps, axis=1); thr = np.quantile(gmin, 0.2)
    for j in range(1, len(times)):
        if gmin[j] <= thr: ax.axvspan(times[j-1], times[j], alpha=0.12)
    ax.set_xlabel("time"); ax.set_ylabel("λ_k"); ax.set_title("Spectral braid"); stamp(ax, stamp_text, 'lr'); fig.tight_layout()
    if out: fig.savefig(out, dpi=150); plt.close(fig); return fig

def continuity_waterfall(cont_scores: np.ndarray, stamp_text=None, out=None):
    Tm1, K = cont_scores.shape; fig, ax = plt.subplots(figsize=(8, 1 + 0.4*K))
    for k in range(K): ax.plot(np.arange(Tm1), k + cont_scores[:,k], lw=1.2)
    ax.set_yticks(np.arange(K)); ax.set_yticklabels([f"k={i+1}" for i in range(K)]); ax.set_xlabel("frame"); ax.set_title("Continuity waterfall |<φ_k(t), φ_k(t+Δt)>|")
    stamp(ax, stamp_text, 'lr')
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return fig
    return fig

def chord_from_matrix(M: np.ndarray, title: str, stamp_text=None, out=None, max_edges=60):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Circle
    K = M.shape[0]; V = np.abs(M.copy()); np.fill_diagonal(V, 0.0)
    tri = np.transpose(np.nonzero(np.triu(np.ones_like(V), k=1)))
    edges = [(i,j,V[i,j]) for (i,j) in tri]; edges.sort(key=lambda x: x[2], reverse=True); edges = edges[:min(max_edges, len(edges))]
    fig, ax = plt.subplots(figsize=(6,6)); ang = np.linspace(0, 2*np.pi, K, endpoint=False); xy = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for i,(x,y) in enumerate(xy):
        circ = Circle((x,y), 0.04, fill=True); ax.add_patch(circ); ax.text(x, y+0.08, f"{i+1}", ha="center", va="bottom")
    for (i,j,w) in edges:
        x0,y0 = xy[i]; x1,y1 = xy[j]; path = Path([(x0,y0),(0.0,0.0),(x1,y1)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        patch = PathPatch(path, lw=0.8 + 2.5*w/np.max(V+1e-16), alpha=0.5, fill=False); ax.add_patch(patch)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title); stamp(ax, stamp_text, 'lr'); fig.tight_layout()
    if out: fig.savefig(out, dpi=150); plt.close(fig); return fig

def holonomy_phase_wheel(diagP: np.ndarray, stamp_text=None, out=None):
    fig, ax = plt.subplots(figsize=(4,4))
    ang = np.angle(diagP); rad = np.abs(diagP)
    t = np.linspace(0, 2*np.pi, 256); ax.plot(np.cos(t), np.sin(t), lw=1.0)
    ax.scatter(np.cos(ang)*rad, np.sin(ang)*rad)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title("Holonomy phase wheel"); stamp(ax, stamp_text, 'lr'); fig.tight_layout()
    if out: fig.savefig(out, dpi=150); plt.close(fig); return fig

def holonomy_perm_heat(P: np.ndarray, stamp_text=None, out=None):
    fig, ax = plt.subplots(figsize=(5,4)); ax.imshow(np.abs(P), origin='lower', interpolation='nearest')
    K = P.shape[0]; argmax = np.argmax(np.abs(P), axis=1); ax.plot(np.arange(K), argmax, marker='o', lw=0.0)
    ax.set_title("|P| with row argmax"); stamp(ax, stamp_text, 'lr'); fig.tight_layout()
    if out: fig.savefig(out, dpi=150); plt.close(fig); return fig
