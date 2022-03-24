import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner_by_sum, get_bpx_preconditioner


L = 10
C_naive = get_bpx_preconditioner_by_sum(L)
C_analytical = get_bpx_preconditioner(L)

C_naive_eval = C_naive.eval(reshape='matrix')
C_analytical_eval = C_analytical.eval(reshape='matrix')
fig, axes = plt.subplots(1, 3, dpi=100, figsize=(14,9))

rel_error = np.sqrt(np.sum((C_naive_eval - C_analytical_eval)**2) / np.sum(C_analytical_eval**2))
print(f"rel error: {rel_error:.2e}")

axes[0].imshow(C_naive_eval)
axes[1].imshow(C_analytical_eval)
axes[2].imshow(C_naive_eval - C_analytical_eval)

