import pint
from matplotlib import pyplot as plt
import numpy as np

ureg = pint.UnitRegistry()

g_C = (2.4e-3 * ureg.millielectron_volt * ureg.micrometer * ureg.micrometer)
g_R = 2.0 * g_C
R = (1.2e-3 * ureg.millieV * ureg.micrometer**2 * ureg.hbar**-1)
gamma_C = (0.2 * ureg.picosecond ** -1)
a = 1.1
gamma_R = a * gamma_C
r = (5.5 * ureg.micrometer)
m = 7e-5 * ureg.electron_mass
sigmaP = (1.0 * ureg.micrometer)
Pth = gamma_C * gamma_R / R

Ps = np.linspace(0.1, 3, num=1000)

bs1 = ((ureg.hbar / 2) * ((2 * Pth * Ps * g_R) / (m * r**2 * gamma_R))**0.5).\
    to("millieV").magnitude
bs2 = ((2 * g_C * gamma_R / R) * (1 - (1 / np.sqrt(2 * np.pi) * (r / sigmaP) \
                                     * (1 / Ps)))).to("millieV").magnitude

p1 = plt.plot(Ps, bs1)
p2 = plt.plot(Ps, bs2)
p3 = plt.plot(Ps, bs1 + bs2)
plt.gca().set_xlabel("P/Pth")
plt.gca().set_ylabel("Blueshift (meV)")
plt.gca().set_title("Blueshift vs Power (Harmonic Contribution)")
plt.legend([p1, p2, p3], ['Harmonic Contribution',
                          'Nonlinear Contribution',
                          "Blueshift"])
plt.show()
