import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_bpx_Qt, get_overlap_matrix_as_tt, _get_refinement_tensor
import laplace_mps.tensormethods as tm

L = 3
Qt = get_bpx_Qt(L)
M = get_overlap_matrix_as_tt(L)

plt.close("all")
fig, axes = plt.subplots(2, 2, dpi=100, figsize=(14,9))
fig.suptitle("Overlap Matrix")
indices = list(range(0, 2*L, 2)) + list(range(1,2*L, 2)) + [-1]
M_eval = M.eval().transpose(indices).reshape([2**L, 2**L, 2])
for i in range(2):
    axes[0][i].imshow(M_eval[:,:,i])


# CMt_eval = CMt.eval().transpose(indices).reshape([2**L, 2**L, 2])
# Qt_eval = Qt.eval().transpose(indices).reshape([2**L, 2**L, 2])
#
# for i in range(2):
#     axes[i][0].imshow(CMt_eval[:, :, i])
#     axes[i][1].imshow(Qt_eval[:, :, i])
#     axes[i][2].imshow(CMt_eval[:, :, i] - Qt_eval[:, :, i])
#
# axes[0][0].set_title("Naive")
# axes[0][1].set_title("Q directly")
# axes[0][2].set_title("Residual")