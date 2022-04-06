from sympy import sin, cos, symbols, diff, integrate, pi

x,y = symbols("x y")

f = -3*x**3 + 5*x**2 - x
g = (2*y**2 - 3*y + 1) * sin(2*pi*y)

f_L2 = integrate(f**2, (x, 0, 1))
g_L2 = integrate(g**2, (y, 0, 1))
g_H1 = integrate(diff(g, y)**2, (y, 0, 1))

f1 = diff(f, x, 1).simplify()
g1 = diff(g, y, 1).simplify()


f2 = diff(f, x, 2).simplify()
g2 = diff(g, y, 2).simplify()
rhs = (diff(f*g, x, 2) + diff(f*g, y, 2)).simplify()

fg_L2 = integrate((f*g)**2, (x, 0, 1), (y,0,1))
fg_H1 = integrate((g*f2+f*g2)*f*g, (x, 0, 1), (y,0,1)).simplify()

print(fg_H1)

print("|g(y)| L2: ", g_L2)
print("|g(y)| H1: ", g_H1)


