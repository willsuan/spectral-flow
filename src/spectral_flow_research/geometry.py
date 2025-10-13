
import numpy as np
from scipy.ndimage import distance_transform_edt

def signed_distance(mask: np.ndarray, h: float) -> np.ndarray:
    """Signed distance: positive outside, negative inside; scaled by h."""
    inside = mask.astype(bool); outside = ~inside
    d_in = distance_transform_edt(inside); d_out = distance_transform_edt(outside)
    return (d_out - d_in) * h

def gradient_central(u: np.ndarray, h: float):
    """Second-order central differences with edge one-sided fallbacks."""
    ux = np.zeros_like(u); uy = np.zeros_like(u)
    ux[1:-1,:] = (u[2:,:] - u[:-2,:])/(2*h); uy[:,1:-1] = (u[:,2:] - u[:,:-2])/(2*h)
    ux[0,:] = (u[1,:]-u[0,:])/h; ux[-1,:] = (u[-1,:]-u[-2,:])/h
    uy[:,0] = (u[:,1]-u[:,0])/h; uy[:,-1] = (u[:,-1]-u[:,-2])/h
    return ux, uy

def delta_kernel(s: np.ndarray, sigma: float):
    """Gaussian delta with width sigma."""
    return (1.0/(np.sqrt(2.0*np.pi)*sigma)) * np.exp(-0.5*(s/sigma)**2)
