import sympy
sympy.init_printing()

t, s = sympy.symbols('t, s')
a = sympy.symbols('a', real=True, positive=True)

f = sympy.exp(-a*t)
f2 = f.subs({a:2})

 p = sympy.plot(f2, sympy.Heaviside(t),
                xlim=(-1, 4), ylim=(0, 3), show=True)

