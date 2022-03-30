import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner_by_sum, get_bpx_preconditioner, get_bpx_Qp, get_bpx_Qt, get_derivative_matrix_as_tt, get_overlap_matrix_as_tt, get_gram_matrix_as_tt, get_laplace_matrix_as_tt


L = 3
C = get_bpx_preconditioner(L)
Qp = get_bpx_Qp(L)
Qt = get_bpx_Qt(L)
D = get_derivative_matrix_as_tt(L)
M = get_overlap_matrix_as_tt(L)
C_expanded = C.copy()
C_expanded.tensors.append(np.eye(2)[None, :, :, None])
CMt = C_expanded @ M.copy().transpose()
G = get_gram_matrix_as_tt(L)

mass_naive = (CMt @ G @ CMt.copy().transpose()).squeeze()
mass_directly = (Qt @ G @ Qt.copy().transpose()).squeeze()



plt.close("all")
fig, axes = plt.subplots(1, 3, dpi=100, figsize=(14,9))
fig.suptitle("Derivative Matrix")
rel_error = np.sqrt(np.sum((Qp.evalm() - (D@C).evalm())**2) / np.sum(Qp.evalm() **2))
print(f"rel error derivative matrix: {rel_error:.2e}")
axes[0].imshow((D@C).evalm())
axes[1].imshow(Qp.evalm())
axes[2].imshow(Qp.evalm() - (D@C).evalm())
axes[0].set_title("Naive")
axes[1].set_title("Q directly")
axes[2].set_title("Residual")


fig, axes = plt.subplots(2, 3, dpi=100, figsize=(14,9))
fig.suptitle("Overlap Matrix")
indices = list(range(0, 2*L, 2)) + list(range(1,2*L, 2)) + [-1]
CMt_eval = CMt.eval().transpose(indices).reshape([2**L, 2**L, 2])
Qt_eval = Qt.eval().transpose(indices).reshape([2**L, 2**L, 2])

for i in range(2):
    axes[i][0].imshow(CMt_eval[:, :, i])
    axes[i][1].imshow(Qt_eval[:, :, i])
    axes[i][2].imshow(CMt_eval[:, :, i] - Qt_eval[:, :, i])

axes[0][0].set_title("Naive")
axes[0][1].set_title("Q directly")
axes[0][2].set_title("Residual")

rel_error = np.sqrt(np.sum((CMt_eval - Qt_eval)**2) / np.sum(Qt_eval**2))
print(f"rel error overlap matrix: {rel_error:.2e}")



# C_naive = get_bpx_preconditioner_by_sum(L)
# C_analytical = get_bpx_preconditioner(L)
#
# C_naive_eval = C_naive.eval(reshape='matrix')
# C_analytical_eval = C_analytical.eval(reshape='matrix')
# fig, axes = plt.subplots(1, 3, dpi=100, figsize=(14,9))
#
# rel_error = np.sqrt(np.sum((C_naive_eval - C_analytical_eval)**2) / np.sum(C_analytical_eval**2))
# print(f"rel error: {rel_error:.2e}")
#
# axes[0].imshow(C_naive_eval)
# axes[1].imshow(C_analytical_eval)
# axes[2].imshow(C_naive_eval - C_analytical_eval)

