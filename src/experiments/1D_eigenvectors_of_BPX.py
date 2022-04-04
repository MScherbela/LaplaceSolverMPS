import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner

L = 7
C = get_bpx_preconditioner(L).evalm()

eigvals, u = np.linalg.eigh(C)
n_eigvalues = 5

plt.close("all")
fig, axes = plt.subplots(n_eigvalues, 2, dpi=100, figsize=(14,9))
for k in range(n_eigvalues):
    axes[k][0].plot(u[:,k], label=f"$\\lambda$ = {eigvals[k]:.4f}", color='C0')
    axes[k][1].plot(u[:,2**L-k-1], label=f"$\\lambda$ = {eigvals[-k]:.4f}", color='C1')

axes[0][0].set_title("Smallest eigenvalues")
axes[0][1].set_title("Largest eigenvalues")



