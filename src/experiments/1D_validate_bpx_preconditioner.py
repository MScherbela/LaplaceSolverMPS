import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner, get_bpx_Qp, get_bpx_Qt, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt, get_gram_matrix_as_tt

L = 4
C = get_bpx_preconditioner(L)
Qp = get_bpx_Qp(L)
Qt = get_bpx_Qt(L)
D = get_derivative_matrix_as_tt(L)
M = get_overlap_matrix_as_tt(L)
C_expanded = C.copy()
C_expanded.tensors.append(np.ones([1,1,1,1]))
CMt = C_expanded @ M.copy().transpose()
G = get_gram_matrix_as_tt(L)

mass_naive = (CMt @ G @ CMt.copy().transpose()).squeeze()
mass_directly = (Qt @ G @ Qt.copy().transpose()).squeeze()


plt.close("all")
fig, axes = plt.subplots(1, 3, dpi=100, figsize=(12,7))
fig.suptitle("Derivative Matrix")
rel_error = np.sqrt(np.sum((Qp.evalm() - (D@C).evalm())**2) / np.sum(Qp.evalm() **2))
print(f"rel error derivative matrix: {rel_error:.2e}")
axes[0].imshow((D@C).evalm())
axes[1].imshow(Qp.evalm())
axes[2].imshow(Qp.evalm() - (D@C).evalm())
axes[0].set_title("Naive")
axes[1].set_title("Q directly")
axes[2].set_title("Residual")


fig, axes = plt.subplots(2, 3, dpi=100, figsize=(12,7))
fig.suptitle("Preconditioned overlap matrix $Q^T = CM^T$", fontsize=14)
indices = list(range(0, 2*L, 2)) + list(range(1,2*L, 2)) + [-1]
CMt_eval = CMt.eval().transpose(indices).reshape([2**L, 2**L, 2])
Qt_eval = Qt.eval().transpose(indices).reshape([2**L, 2**L, 2])

for i in range(2):
    axes[i][0].imshow(CMt_eval[:, :, i], clim=[-0.3, 0.3], cmap='bwr')
    axes[i][1].imshow(Qt_eval[:, :, i], clim=[-0.3, 0.3], cmap='bwr')
    axes[i][2].imshow((CMt_eval[:, :, i] - Qt_eval[:, :, i])*1e14, clim=[-0.5, 0.5], cmap='bwr')

axes[0][0].set_title("Naive: Matrix product $CM$")
axes[0][1].set_title("Q assembled directly")
axes[0][2].set_title("Residual * $10^{14}$")
axes[0][0].set_ylabel("Q[:,:,0]")
axes[1][0].set_ylabel("Q[:,:,1]")
fig.tight_layout()
fig.savefig("outputs/1D_validate_Q.pdf", bbox_inches='tight')

rel_error = np.sqrt(np.sum((CMt_eval - Qt_eval)**2) / np.sum(Qt_eval**2))
print(f"rel error overlap matrix: {rel_error:.2e}")