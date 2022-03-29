import tt
from tt.amen import amen_solve
import laplace_mps.tensormethods as tm
from laplace_mps.solver import get_laplace_matrix_as_tt, solve_with_grad_descent
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

L = 3
tensors = [np.random.normal(size=[2,2,2,2]) for _ in range(L)]
tensors[0] = tensors[0][:1, ...]
tensors[-1] = tensors[-1][..., :1]
A = tm.TensorTrain(tensors)
A = A.copy().transpose() @ A

tensors = [np.random.normal(size=[2,2,2]) for _ in range(L)]
tensors[0] = tensors[0][:1, ...]
tensors[-1] = tensors[-1][..., :1]
b = tm.TensorTrain(tensors)

A_eval = A.evalm()
b_eval = b.evalv()

x_dense = np.linalg.solve(A_eval, b_eval)
x_tm, _ = solve_with_grad_descent(A, b, print_steps=True, n_steps_max=1000)
x_tm_eval = x_tm.evalv()

A_ttpy = tt.matrix.from_list(A.to_ttpylist())
b_ttpy = tt.vector.from_list(b.to_ttpylist())
x0_ttpy = tt.ones(2, L)
x_ttpy = amen_solve(A_ttpy, b_ttpy, x0_ttpy, eps=1e-14)
x_ttpy_eval = tm.TensorTrain.from_ttpylist(tt.vector.to_list(x_ttpy)).evalv()

plt.close("all")
plt.figure(dpi=100)
plt.plot(x_dense)
plt.plot(x_tm_eval, ls='--')
plt.plot(x_ttpy_eval, ls=':')







