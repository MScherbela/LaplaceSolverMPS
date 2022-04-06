import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner

L = 7
C = get_bpx_preconditioner(L).evalm()

eigvals, u = np.linalg.eigh(C)
n_eigvalues = 3

plt.close("all")
fig, axes = plt.subplots(n_eigvalues, 2, dpi=100, figsize=(9,6), sharex=True)
for k in range(n_eigvalues):
    axes[k][0].plot(u[:,k], label=f"$\\lambda$ = {eigvals[k]:.4f}", color='C0')
    axes[k][1].plot(u[:,2**L-k-1], label=f"$\\lambda$ = {eigvals[-k]:.4f}", color='C1')

for ax in axes:
    for a in ax:
        a.grid(alpha=0.5)

fig.suptitle("Eigenvectors of $C$ corresponding to ...", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.87)
axes[0][0].set_title("... smallest eigenvalues")
axes[0][1].set_title("... largest eigenvalues")

fig.savefig("outputs/1D_BPX_eigenvectors.pdf", bbox_inches='tight')



