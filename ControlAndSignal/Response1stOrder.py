#https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Book%3A_Introduction_to_Linear_Time-Invariant_Dynamic_Systems_for_Students_of_Engineering_(Hallauer)/04%3A_Frequency_Response_of_First_Order_Systems_Transfer_Functions_and_General_Method_for_Derivation_of_Frequency_Response/4.02%3A_Response_of_a_First_Order_System_to_a_Suddenly_Applied_Cosine
import sympy
print(sympy.init_printing())

'''
t, s = sympy.symbols('t, s')
a = sympy.symbols('a', real=True, positive=True)

f = sympy.exp(-a*t)
print(f)

F = sympy.integrate(f*sympy.exp(-s*t), (t, 0, sympy.oo))
print(F.subs({a:2}))

def laplace(fct, noconds = True):
    return sympy.laplace_transform(fct, t, s, noconds=noconds)

print(laplace(f, noconds=False))
print(laplace(f))

'''
