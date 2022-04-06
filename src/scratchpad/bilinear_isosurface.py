import numpy as np
import matplotlib.pyplot as plt


def hat_func(x):
    return np.where(x < 1, x, 2-x)

x = np.linspace(0, 2.0, 50)
y = np.linspace(0, 2.0, 50)
xx,yy = np.meshgrid(x,y)
zz = hat_func(xx) * hat_func(yy)

plt.close("all")
plt.contour(zz, levels=20)
plt.axis("equal")

