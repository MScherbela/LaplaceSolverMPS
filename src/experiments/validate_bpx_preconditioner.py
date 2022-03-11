import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.solver import get_BPX_preconditioner_naive, get_bpx_preconditioner


L = 10
C_naive = get_BPX_preconditioner_naive(L)
C_analytical = get_bpx_preconditioner(L)

C_naive_eval = C_naive.eval(reshape='matrix')
C_analytical_eval = C_analytical.eval(reshape='matrix')
fig, axes = plt.subplots(1, 3, dpi=100, figsize=(14,9))

axes[0].imshow(C_naive_eval)
axes[1].imshow(C_analytical_eval)
axes[2].imshow(C_naive_eval - C_analytical_eval)

