"""
To produce plots of dependence of blueshift on intensity and pump power, and
dependence of intnesity on pump power
"""
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import h5py
from os import path
from UtilityFunctions import loadGPESolver, diagnosticPlot, getPlotName

FOLDER_NAME = "PScanLiew R=0.0012,g_C=0.0024,gamma_C=0.2,gamma_R=0.22,m=7e-05"
PATH_TO_FILES = path.join(path.expanduser("~/"), "Dropbox", "Daniel_Honours",
                          "Numerics", FOLDER_NAME)
FILE_NAME = "PScan"

def loadAndPlotSolver(f):
    """
    Load a solver from file f and produce the diagnostic plot

    Parameters:
        f: h5py.File instance
    Returns:
    """
    solver = loadGPESolver(f)
    P = f.attrs["P"]
    pl = diagnosticPlot(solver)
    fname = path.join(PATH_TO_FILES, FILE_NAME
                    + getPlotName(solver.paramContainer.getOutputParams(), P=P)
                    + ".png")
    pl.savefig(fname, dpi=500)
    plt.close(pl)
    plt.close()

fNames = glob(PATH_TO_FILES + "/*.hdf5")

for fName in fNames:
    f = h5py.File(fName, "r")
    loadAndPlotSolver(f)
    f.close()
