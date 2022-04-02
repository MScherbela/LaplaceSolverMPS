import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_preconditioner_by_sum_2D, _get_bpx_factors, _get_refinement_tensor


L = 3
# C_sum = get_bpx_preconditioner_by_sum_2D(L)
U, V, W, Y = _get_bpx_factors()
U_hat = _get_refinement_tensor()

plt.close("all")
plt.figure(figsize=(14,8), dpi=100)
plt.imshow(U.transpose([0,1,3,2]).reshape([8,8]))
plt.axhline(3.5, color='w')
plt.axvline(3.5, color='w')






#%%
C_sum_eval = C_sum.evalm()
C_eval = C.evalm()
residual = C_eval - C_sum_eval
fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
axes[0][0].imshow(C_sum_eval)
axes[0][0].set_title("Naive C (as sum)")

axes[0][1].imshow(C_eval)
axes[0][1].set_title("TT 2D C")

axes[1][0].imshow(residual)
axes[1][0].set_title("Residual")

axes[1][1].imshow(np.log2(C_eval / C_sum_eval), clim=[-2,2], cmap='bwr')
axes[1][1].set_title("log2(naive/sum)")