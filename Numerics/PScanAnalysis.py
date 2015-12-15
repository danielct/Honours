"""
To produce plots of dependence of blueshift on intensity and pump power, and
dependence of intnesity on pump power
"""
# TODO: also extract blushift from dispersion
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import h5py
from os import path
from pint import UnitRegistry

ur = UnitRegistry()
PLOT_NAME = "Analysis"
FOLDER_NAME = "PScanLiew R=0.0012,g_C=0.0024,gamma_C=0.2,gamma_R=0.22,m=7e-05"
PATH_TO_FILES = path.join(path.expanduser("~/"), "Dropbox", "Daniel_Honours",
                          "Numerics", FOLDER_NAME)
# P_CUTOFF = 1.6
P_CUTOFF = 1.6


def getVals(f):
    """
    Returns the intensity, mean field enery, and pump power associated with a
    saved GPESolver.

    Parameters:
        f: h5py.File instance
    Returns:
        Power, Number, Energy
    """
    P = f.attrs["P"]
    N = f["n"].shape[0]
    energy = f["energy"][-1]
    spectrum = np.absolute(np.array(f["spectrum"]))
    nomega, nk = spectrum.shape
    omega_axis = f["omega_axis"] / ur.Quantity(f.attrs["charT"],
                                                 ur.picosecond.units)
    # Energy axis in milliev
    energy_axis = (omega_axis * ur.hbar).to("millieV").magnitude

    blueshiftind = np.argmax(np.array(spectrum[:, nk//2]))
    blueshift = energy_axis[blueshiftind]
    number = f["number"][-1]
    profile = np.absolute(np.array(f["psi"][N/2, :]))

    return (P, energy, blueshift, number, profile)

fNames = glob(PATH_TO_FILES + "/*.hdf5")

f = h5py.File(fNames[0], "r")
x_ax = np.array(f["x_ax"]) * f.attrs["charL"] * 1e6
N = x_ax.shape[0]
f.close()

numFiles = len(fNames)
Ps = np.zeros(numFiles)
numbers = np.zeros_like(Ps)
blueshifts = np.zeros_like(Ps)
energies = np.zeros_like(Ps)
profiles = np.zeros((numFiles, N))


for i, fname in enumerate(fNames):
    f = h5py.File(fname, "r")
    l = fname.rfind("=") + 1
    u = fname.rfind(".")
    P = float(fname[l:u])
    if P < P_CUTOFF:
        Ps[i], energies[i], blueshifts[i], numbers[i], profiles[i, :]\
            = getVals(f)
        print getVals(f)[0]
        # Ps[i] = float(fname[l:u])
        print Ps[i]
    f.close()

fig, axes = plt.subplots(2, 2)

axes[0, 0].semilogy(Ps, numbers, "o")
axes[0, 0].set_xlabel("P / Pth")
axes[0, 0].set_ylabel("Number")
axes[0, 0].set_title("Number vs Power")

for profile in profiles:
    axes[0, 1].hold(True)
    axes[0, 1].plot(x_ax, profile)
axes[0, 1].set_xlim([-5, 5])
axes[0, 1].set_xlabel("r (micron)")
axes[0, 1].set_ylabel("Number")
axes[0, 1].set_title("Center Line Profile")

axes[1, 0].plot(Ps, energies, color="black", marker="o", ls="")
axes[1, 0].set_xlabel("P / Pth")
axes[1, 0].set_ylabel("Mean Field Energy (au)")
ax2 = axes[1, 0].twinx()
ax2.plot(Ps, blueshifts, 'b', marker=".", ls='')
ax2.set_ylabel("Blueshift (meV)", color='b')
for t1 in ax2.get_yticklabels():
    t1.set_color('b')

axes[1, 1].semilogx(numbers, energies, color="black", marker="o", ls="")
axes[1, 1].set_xlabel("Number")
axes[1, 1].set_ylabel("Mean Field Energy (au)")
ax2 = axes[1, 1].twinx()
ax2.semilogx(numbers, 1e3*blueshifts, 'b', marker=".", ls='')
ax2.set_ylabel("Blueshift (meV)", color='b')
for t1 in ax2.get_yticklabels():
    t1.set_color('b')

fig.tight_layout()
fig.savefig(path.join(PATH_TO_FILES, PLOT_NAME))
plt.close(fig)


# name = fNames[6]
# print name
# f = h5py.File(name, "r")
# print "\n" + str(f.attrs["P"])
# spectrum = np.absolute(np.array(f["spectrum"]))
# nomega, nk = spectrum.shape
# omega_axis = f["omega_axis"] / ur.Quantity(f.attrs["charT"],
#                                              ur.picosecond.units)
# # Energy axis in milliev
# energy_axis = (omega_axis * ur.hbar).to("millieV").magnitude
# print energy_axis
# plt.plot(energy_axis, spectrum[:, nk//2])
# plt.gca().set_xlim([-4, 4])
# yu, yl = plt.gca().get_ylim()
# #plt.gca().set_ylim([-0.1*yu, yu])
# plt.show()
