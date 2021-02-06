#https://dynamics-and-control.readthedocs.io/en/latest/1_Dynamics/3_Linear_systems/Laplace%20transforms.html
import sympy
sympy.init_printing()

#import matplotlib.pyplot as plt

t, s = sympy.symbols('t, s')
a = sympy.symbols('a', real=True, positive=True)

f = sympy.exp(-a*t)
print(f)

F = sympy.integrate(f*sympy.exp(-s*t), (t, 0, sympy.oo))
print(F.subs({a:2}))

def L(f, noConds = True):
    return sympy.laplace_transform(f, t, s, noconds=noConds)

print(L(f, noConds=False))
print(L(f))

