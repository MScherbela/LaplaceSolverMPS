import numpy as np

N = 100
x0 = np.random.normal(1.0, 1.0, size=N)
# dx = 1e-8 * x0 # fully correlated errors
dx = np.random.normal(0, 1.0, size=N) * 1e-8
x = x0 + dx

rel_error = np.linalg.norm(x0 - x)
norm_x0 = np.linalg.norm(x0)
norm_x = np.linalg.norm(x)
rel_error_of_norms = np.abs(norm_x - norm_x0) / norm_x0

print(f"Rel error             |u-u0|  / |u0|: {rel_error:.2e}")
print(f"Rel error of norms (|u|-|u0|) / |u0|: {rel_error_of_norms:.2e}")



