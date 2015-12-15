import pint
import sympy
ureg = pint.UnitRegistry()

a = 1.5
hbar = ureg.hbar.to_base_units()
gamma_C = (200 * ureg.picosecond) ** (-1)
gamma_C = gamma_C.to_base_units()
gamma_R = 4.0 * (0.5 * ureg.millielectron_volt) / hbar
gamma_R = gamma_R.to_base_units()
R = (0.05 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer) / hbar
R = R.to_base_units()

g_C = (2.0 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer).\
    to_base_units()
g_R = 2.0 * g_C
m = 1e-4 * ureg.electron_mass.to_base_units().magnitude

L = 2 * (80 * ureg.micrometer).to_base_units().magnitude

V = (1 / R) * (g_R * gamma_C + g_C * gamma_R * (a - 1))
# print V.dimensionality
V = V.to_base_units().magnitude

V, L, hBar, m = sympy.symbols('V L hbar.magnitude m')
