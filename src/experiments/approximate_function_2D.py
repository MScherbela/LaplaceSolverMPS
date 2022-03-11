import matplotlib.pyplot as plt
from laplace_mps.utils import get_u_function_as_tt, TrigFunction
from numpy.polynomial import Polynomial
import numpy as np

L = 5

poly_x, trig_x = Polynomial([1,-4,4]), [TrigFunction(1,0,0)]
poly_y, trig_y = Polynomial([1]), [TrigFunction(1,0,5*np.pi)]

fx = get_u_function_as_tt(poly_x, trig_x, L)
fy = get_u_function_as_tt(poly_y, trig_y, L)

f = fx.expand_dims(1) @ fy.expand_dims(0)


f_eval = f.eval(reshape='matrix')
f_eval_rr = f_eval[1::2, 1::2]

plt.close("all")
plt.imshow(f_eval_rr, cmap='bwr', clim=np.array([-1,1]) * np.max(f_eval_rr))




