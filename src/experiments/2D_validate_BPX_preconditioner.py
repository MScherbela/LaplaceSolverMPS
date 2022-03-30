import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import build_laplace_matrix_2D, get_bpx_preconditioner_by_sum_2D, get_laplace_bpx_2D, get_rhs_matrix_as_tt_2D, get_bpx_Qt_term, get_gram_matrix_as_tt
import laplace_mps.tensormethods as tm

def get_R2D_bpx_naive(C):
    L = len(C)
    C_expanded = C.copy()
    C_expanded.tensors.append(np.eye(4)[None, :, :, None])
    return C_expanded @ get_rhs_matrix_as_tt_2D(L)

rel_error = 1e-13
max_rank = 80

L = 3

# LHS
B = get_laplace_bpx_2D(L)
A = build_laplace_matrix_2D(L).reapprox(rel_error=rel_error)
C = get_bpx_preconditioner_by_sum_2D(L).reapprox(rel_error=rel_error)
B_naive = (C @ A @ C).reapprox(rel_error=rel_error)
B_naive.reshape_mode_indices([4,4])
rel_error = np.sqrt((B-B_naive).flatten_mode_indices().norm_squared() / B.norm_squared())
print(f"rel error: {rel_error:.1e}")

#%%
# RHS
R_2D_bpx_naive = get_R2D_bpx_naive(C)
R_2D_bpx = get


plt.close("all")
indices = list(range(0, 2*L, 2)) + list(range(1, 2*L+1, 2)) + [-1]
R_2D_bpx_naive_eval = R_2D_bpx_naive.eval().transpose(indices).reshape([4**L, 4**L, 4])
R_2D_bpx_eval = R_2D_bpx.eval().transpose(indices).reshape([4**L, 4**L, 4])
residual = R_2D_bpx_naive_eval - R_2D_bpx_eval
ratio = R_2D_bpx_naive_eval / R_2D_bpx_eval
fig, axes = plt.subplots(4, 4, dpi=100, figsize=(14,10))
for i in range(4):
    axes[0][i].imshow(R_2D_bpx_naive_eval[:,:,i])
    axes[1][i].imshow(R_2D_bpx_eval[:,:,i])
    axes[2][i].imshow(residual[:,:,i])
    axes[3][i].imshow(np.log2(ratio[:, :, i]), clim=[-3,3], cmap='bwr')


#%%
B_naive_eval = B_naive.evalm()
B_eval = B.evalm()
residual = B_naive_eval - B_eval
fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
axes[0][0].imshow(B_naive_eval)
axes[0][0].set_title("Naive (CAC)")

axes[0][1].imshow(B_eval)
axes[0][1].set_title("Sum")

axes[1][0].imshow(residual)
axes[1][0].set_title("Residual")

axes[1][1].imshow(np.log2(B_eval / B_naive_eval), clim=[-2,2], cmap='bwr')
axes[1][1].set_title("log2(naive/sum)")



print(f"rel error: {np.sqrt(np.sum(residual**2) / np.sum(B_eval**2)):.1e}")


