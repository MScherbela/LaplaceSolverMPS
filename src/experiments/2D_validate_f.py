import numpy as np
import matplotlib.pyplot as plt
from laplace_mps.utils import get_example_u_2D, get_example_f_2D

def get_naive_laplace(u):
    f = 4 * u
    f -= np.roll(u, shift=1, axis=0)
    f -= np.roll(u, shift=-1, axis=0)
    f -= np.roll(u, shift=1, axis=1)
    f -= np.roll(u, shift=-1, axis=1)

    f[:,0] = 0
    f[:, -1] = 0
    f[0, :] = 0
    f[-1, :] = 0
    return f

def imshow(ax, x):
    ax.imshow(x, cmap='bwr', clim=np.array([-1, 1]) * np.max(np.abs(x)), origin='lower', extent=[0, 1, 0, 1])


L = 6
u = get_example_u_2D(L, basis='nodal')
f = get_example_f_2D(L)

u_eval = u.reshape_mode_indices([2, 2]).evalm()
f_eval = f.evalm()[1::2, :][:, 1::2]
f_naive = get_naive_laplace(u_eval)



plt.close("all")
fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
imshow(axes[0][0], u_eval.T)
axes[0][0].set_title("Reference solution")

imshow(axes[0][1], f_eval.T)
axes[0][1].set_title("f from TT")

u_slice = u_eval[2**L // 2, :]
f_slice = f_eval[2**L // 2, :]
axes[1][0].plot(np.linspace(0, 1, len(u_slice)), u_slice)
# axes[1][0].plot(np.diff(u_slice))
axes[1][0].plot(np.linspace(0, 1, len(u_slice)-2), -np.diff(np.diff(u_slice)) * 10)
axes[1][0].plot(np.linspace(0, 1, len(u_slice)), f_slice/10)

imshow(axes[1][1], f_naive.T)
axes[1][1].set_title("f from num. diff")
#
# axes[1][0].imshow(u_sol_eval.T, cmap='bwr', clim=np.array([-1, 1]) * np.max(u_sol_eval), origin='lower', extent=[0, 1, 0, 1])
# axes[1][0].set_title("PDE solution")

for ax in axes:
    for a in ax:
        a.grid(alpha=0.2)
