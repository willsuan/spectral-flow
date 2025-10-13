
from spectral_flow_research.experiment import run
if __name__ == "__main__":
    rd = run(n=128, K=16, frames=36, T=0.6, shape_from="ellipse", shape_to="cardioid",
             gauge="polar", connection_mode="log_polar", equal_area=True, seed=2025)
    print("Demo complete. See:", rd)
