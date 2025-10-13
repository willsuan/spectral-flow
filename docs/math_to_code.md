
# Math → Code mapping (high level)

- **Eigen-tracking:** weighted Gram–Schmidt (`eigsolve.gram_schmidt`) + matching (`eigsolve.match_modes`) + gauge (`transport.apply_polar_gauge`).
- **Connection:** overlap `connection_overlap(S,dt)`; Lie/log connection `log_connection_from_S(S,dt,mode)`.
- **Curvature:** midpoint (`curvature_midpoint`) and symmetric (Lie) (`symmetric_curvature`).
- **Hadamard:** `hadamard_shape_derivative(_adaptive)` with velocity/acceleration (`boundary_normal_*`).
- **Holonomy:** product (`holonomy_product`) + visuals (`viz.*`).
