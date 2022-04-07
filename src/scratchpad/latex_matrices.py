from laplace_mps.bpx2D import _get_Z0_hat, _get_Z1_hat
import numpy as np

def to_latex_matrix(A, format_func=None, matrix_type='bmatrix', empty_zeros=False):
    if empty_zeros:
        if np.sum(np.abs(A)) == 0:
            return ""
    if format_func is None:
        format_func = lambda x: "{:.0f}".format(x)
    s = f"\\begin{{{matrix_type}}}"
    rows = []
    for row in A:
        rows.append("&".join([format_func(a) for a in row]))
    s += "\\\\".join(rows)
    s += f"\\end{{{matrix_type}}}"
    return s

A = _get_Z1_hat()
# A = _get_Z0_hat()
A = A.transpose([0, 3, 1, 2]) * 4
print(to_latex_matrix(A, lambda x: to_latex_matrix(x, matrix_type='pmatrix', empty_zeros=True)))

