from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pint import UnitRegistry
from numpy import fft
from scipy.ndimage.filters import gaussian_filter
from skimage.restoration import unwrap_phase
import Simulator


def getPlotName(paramCont, **kwargs):
    """
    parameters:
        paramCont: A parameter container instance
    kwargs:
        extra arguments
    """
    outParams = ['R', 'g_C', 'gamma_C', 'gamma_R', 'm'] if paramCont.singleComp\
        else ['g_C', 'gamma_eff', 'gamma_nl', 'm']
    params = paramCont.getOutputParams()
    outStrings = ([u"{0}={1.magnitude:.02}".format(outParam, params[outParam])
                  for outParam in outParams] +
                  ["{}={:.02}".format(argName, argVal)
                   for argName, argVal in kwargs.iteritems()])
    return ",".join(outStrings)


def loadGPESolver(f):
    """
    Load a GPESolver based on the saved solver f
    """
    from pycuda import gpuarray
    ur = UnitRegistry()
    gOutput = "millieV * micrometer**2"
    rOutput = "millieV * micrometer**2 * hbar**-1"
    gammaOutput = "picosecond ** -1"
    gammaNlOutput = "millieV * micrometer**2 * hbar**-1"
    mOutput = "electron_mass"
    charLOutput = "micrometer"
    charTOutput = "picosecond"

    if "gamma_nl" in f.attrs:
        singleComp = True

    g_C = ur.Quantity(f.attrs["g_C"], gOutput)
    g_R = ur.Quantity(f.attrs["g_R"], gOutput)
    gamma_C = ur.Quantity(f.attrs["gamma_C"], gammaOutput)
    gamma_R = ur.Quantity(f.attrs["gamma_R"], gammaOutput)
    R = ur.Quantity(f.attrs["R"], rOutput)
    m = ur.Quantity(f.attrs["m"], mOutput)
    charL = ur.Quantity(f.attrs["charL"], charLOutput)
    charT = ur.Quantity(f.attrs["charT"], charTOutput)

    if singleComp:
        gamma_nl = ur.Quantity(f.attrs["g_R"], gammaNlOutput)
        paramContainer = Simulator.ParameterContainer(g_C=g_C, g_R=g_R, R=R,
                                                      gamma_C=gamma_C,
                                                      gamma_R=gamma_R,
                                                      gamma_nl=gamma_nl,
                                                      m=m, charL=charL,
                                                      charT=charT)
    else:
        paramContainer = Simulator.ParameterContainer(g_C=g_C, g_R=g_R, R=R,
                                                      gamma_C=gamma_C,
                                                      gamma_R=gamma_R,
                                                      m=m, charL=charL,
                                                      charT=charT)
    N = f["n"].shape[0]
    max_XY = f["x_ax"][-1] * charL
    grid = Simulator.Grid(charL.to_base_units().magnitude, N=N, max_XY=max_XY)
    gpeSolver = Simulator.GPESolver(paramContainer, 0.1, grid)
    if singleComp:
        assert gpeSolver.singleComp, "We've made a double comp solver\
        even though we loaded a single one"
    gpeSolver.n = gpuarray.to_gpu(np.array(f["n"],
                                           dtype=gpeSolver.n.get().dtype))
    gpeSolver.Pdt = gpuarray.to_gpu(np.array(f[u'Pdt:'],
                                             dtype=gpeSolver.Pdt.get().
                                             dtype))
    gpeSolver.psi = gpuarray.to_gpu(np.array(f["psi"],
                                             dtype=gpeSolver.psi.get().dtype))
    gpeSolver.spectStartStep = 0
    gpeSolver.spectLength = 0
    gpeSolver.spectEndStep = 0
    if "energy" in f.keys():
        gpeSolver.energy = np.array(f["energy"], dtype=gpeSolver.n.get().dtype)
    if "number" in f.keys():
        gpeSolver.number = np.array(f["number"], dtype=gpeSolver.n.get().dtype)
    if "times" in f.keys():
        gpeSolver.times = np.array(f["times"], dtype=gpeSolver.n.get().dtype)
    if "spectrum" in f.keys():
        gpeSolver.spectrum = np.array(f["spectrum"], dtype=gpeSolver.psi.get().
                                      dtype)
        gpeSolver.omega_axis = np.array(f["omega_axis"], dtype=gpeSolver.n.get()
                                        .dtype)
    return gpeSolver


def scale(inFunction, charL, charOut=1):
    """
    Function to return a scaled version of a spatial function
    Parameters:
        inFunction: function to be scaled
        charL: Characterstic length to scale by
        charOut: Characteristic value of the output of the function

    Returns:
        A scaled version of inFunction. Takes arguments in units of charL, and
        returns output in units of charOut.
    """
    def out(x, y): return inFunction(charL * x, charL * y) / charOut
    return out


def diagnosticPlot(solver):
    """
    Returns a matplotlib figure
    """
    ur = pint.UnitRegistry()
    MASK_CUTOFF = 1.0
    # N_PHASE_TICKS = 6
    NUM_COLOURS = 100
    # DISP_FRACTION = 1/10
    KS_FRACTION = 1/4
    ur = UnitRegistry()
    x, y = solver.grid.getSpatialGrid(scaled=False)
    x *= 1e6
    y *= 1e6

    x_ax = x[0, :]
    y_ax = y[:, 0]
    psi = solver.psi.get()
    # n = solver.n.get()
    P = solver.Pdt.get()
    P /= solver.dt

    N = psi.shape[0]
    time = solver.times
    energy = solver.energy
    number = solver.number
    pumpSlice = P[N/2, :]
    PthScaled = solver.Pth / np.max(P)
    pumpSlice /= np.max(P)

    rMiddle = abs(x[N/2, np.argmax(pumpSlice)])
    density = solver.getDensity(scaled=True)
    pumpCircle = plt.Circle((0.0, 0.0), radius=rMiddle, color='w',
                            linestyle='dashed', fill=False)

    fig, axes = plt.subplots(2, 3)
    time = solver.times
    energy = solver.energy
    number = solver.number

    density = solver.getDensity(scaled=True)
    pumpCircle = plt.Circle((0.0, 0.0), radius=rMiddle, color='w',
                            linestyle='dashed', fill=False)

    fig, axes = plt.subplots(2, 3)

    fig, axes = plt.subplots(2, 3)
    p0 = axes[0, 0].contourf(x, y, density, NUM_COLOURS)

    # RS density
    axes[0, 0].set_title("Density $\mu m ^{-2}$")
    # axes[0, 0].set_aspect('equal', 'box')
    axes[0, 0].set_aspect(1./axes[0, 0].get_data_ratio())
    axes[0, 0].add_artist(pumpCircle)
    plt.colorbar(p0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # KS density
    ksDensity = np.absolute(fft.fftshift(fft.fft2(psi)))
    ksLowerLim = int(N/2 - KS_FRACTION * N / 2)
    ksUpperLim = int(N/2 + KS_FRACTION * N / 2)
    p1 = axes[1, 0].contourf(x_ax[ksLowerLim:ksUpperLim],
                             y_ax[ksLowerLim:ksUpperLim],
                             ksDensity[ksLowerLim:ksUpperLim,
                                       ksLowerLim:ksUpperLim] /
                             np.max(ksDensity), NUM_COLOURS)
    axes[1, 0].set_title("KS Density")
    axes[1, 0].set_aspect('equal')
    plt.colorbar(p1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Phase
    # p2 = axes[0, 1].contourf(x, y, np.angle(psi) + np.pi)
    phase = unwrap_phase(np.angle(psi))
    # Mask the values that are out the damping region are ignored
    mask = solver.damping.get() < MASK_CUTOFF
    phase = np.ma.array(phase, mask=mask) / np.pi
    # Set the minumum phase to zero
    phase -= np.min(phase)
    # maxPhase = np.ceil(np.max(phase) / np.pi)
    # minPhase = np.floor(np.min(phase) / np.pi)
    # step = np.ceil((maxPhase - minPhase) / N_PHASE_TICKS)
    # ticks = np.pi * np.arange(minPhase, maxPhase + 1, step=step)
    # labels = ["$%d \pi$" % i for i in np.arange(minPhase, maxPhase + 1)]
    p2 = axes[0, 1].contourf(x, y, phase, NUM_COLOURS)
    axes[0, 1].set_title("Phase")
    # axes[0, 1].set_aspect('equal', 'box')
    axes[0, 1].set_aspect(1./axes[0, 1].get_data_ratio())
    pumpCircle = plt.Circle((0.0, 0.0), radius=rMiddle, color='w',
                            linestyle='dashed', fill=False)
    axes[0, 1].add_artist(pumpCircle)
    cbar2 = plt.colorbar(p2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    labels = [t.get_text() + '$\pi$' for t in cbar2.ax.get_yticklabels()]
    # cbar2 = plt.colorbar(p2, ax=axes[0, 1],
    #                      ticks=ticks,
    #                      fraction=0.046, pad=0.04)
    cbar2.ax.set_yticklabels(labels)
    # cbar2 = plt.colorbar(p2, ax=axes[0, 1],
    #                      ticks=np.pi * np.array([0, 1.0, 2.0]),
    #                      fraction=0.046, pad=0.04)
    # cbar2.ax.set_yticklabels(['0', '$\pi$', '$2\pi$'])

    # Resovoir density
    # p4 = axes[1, 1].contourf(x, y, n, NUM_COLOURS)
    # axes[1, 1].set_title("Resovoir density")
    # # axes[1, 1].set_aspect('equal', 'box')
    # axes[1, 1].set_aspect(1./axes[1, 1].get_data_ratio())
    # pumpCircle = plt.Circle((0.0, 0.0), radius=rMiddle, color='w',
    #                         linestyle='dashed', fill=False)
    # axes[1, 1].add_artist(pumpCircle)
    # plt.colorbar(p4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Dispersion relation
    # dispLowerLim = int(N/2 - DISP_FRACTION * N / 2)
    # dispUpperLim = int(N/2 + DISP_FRACTION * N / 2)
    # Account for the fact that we have cut off some of the k axis in the
    # spectrum
    spectMax = int(solver.spectrum.shape[1])//2
    k_ax = solver.grid.k_axis_scaled[N//2 - spectMax:N//2 + spectMax]
    # Omega axis in units of 1 / seconds
    omega_axis = solver.omega_axis / ur.Quantity(solver.charT,
                                                 ur.second.to_base_units().
                                                 units)
    # Energy axis in milliev
    energy_axis = (omega_axis * ur.hbar).to("millieV").magnitude
    m = ur.Quantity(solver.paramContainer.getOutputParams()["m"].magnitude,
                    "electron_mass")
    L = ur.Quantity(solver.paramContainer.getOutputParams()["charL"].magnitude,
                    "micrometer")
    dispCurve = ((ur.hbar**2 * k_ax**2) / (2 * m * L**2)).to("millieV")\
        .magnitude
    # p4 = axes[1, 1].contourf(solver.grid.k_axis_scaled[dispLowerLim:
    #                                                    dispUpperLim],
    #                          energy_axis,
    #                          np.absolute(solver.spectrum)[:, dispLowerLim:
    #                                                       dispUpperLim],
    #                          NUM_COLOURS)
    axes[1, 1].contourf(k_ax, energy_axis, np.absolute(solver.spectrum),
                        NUM_COLOURS)
    axes[1, 1].plot(k_ax, dispCurve, color='w', ls=":", lw=0.2)

    yu = 5
    yl = -2
    axes[1, 1].set_ylim([yl, yu])
    # yl = - 1
    # print "yu = %f" % yu
    # print axes[1, 1].get_ylim()
    # axes[1,1].set_ylim([-1, 10])
    # p4 = axes[1, 1].contourf(np.absolute(solver.spectrum)[:, N/4:3*N/4],
    # NUM_COLOURS)
    axes[1, 1].set_title("Dispersion Relation")
    # axes[1, 0].set_aspect('equal')
    axes[1, 1].set_aspect(1./axes[1, 1].get_data_ratio())
    axes[1, 1].set_xlabel('k')
    axes[1, 1].set_ylabel('E (meV)')
    # plt.colorbar(p1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Radial profile
    psiMax = np.max(np.absolute(psi[N/2, :]))
    # axes[0, 2].plot(x[N/2, :], np.absolute(psi[N/2, :]) / psiMax,
    #                 x[N/2, :], pumpFunction(x, y)[N/2, :] / pfMax)

    # axes[0, 2].axhline(PthScaled)
    axes[0, 2].plot(x_ax[N/4:3*N/4], np.absolute(psi[N/2, N/4:3*N/4])/psiMax,
                    x_ax[N/4:3*N/4], pumpSlice[N/4:3*N/4])
    axes[0, 2].set_title("Radial Profile")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_aspect(1./axes[0, 2].get_data_ratio())

    # Energy and number
    axes[1, 2].plot(time, energy, color='k')
    axes[1, 2].ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
    # axes[1, 2].set_title("Energy")
    axes[1, 2].set_xlabel("Time")
    axes[1, 2].set_ylabel("Energy")

    ax2 = axes[1, 2].twinx()
    ax2.plot(time, number, 'b')
    ax2.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
    ax2.set_ylabel("Number", color='b')
    for t1 in ax2.get_yticklabels():
        t1.set_color('b')
    ax2.set_aspect(1./ax2.get_data_ratio())

    # Plot lines to indicate the start and end of spectrum acquisition
    axes[1, 2].axvline(solver.dt * solver.spectStartStep, linestyle='-')
    axes[1, 2].axvline(solver.dt * solver.spectEndStep, linestyle='-')
    axes[1, 2].set_aspect(1./axes[1, 2].get_data_ratio())
    fig.tight_layout()
    return fig

# def diagnosticPlot(x, y, solver, pumpFunction):
#     """
#     Returns a matplotlib figure
#     """
#     psi = solver.psiFinal
#     fig, axes = plt.subplots(2, 4)
#     p0 = axes[0, 0].contourf(x, y, np.absolute(psi))
#     axes[0, 0].set_title("Real Space Density")
#     axes[0, 0].set_aspect('equal')
#     plt.colorbar(p0, ax=axes[0, 0])
#
#     p1 = axes[1, 0].contourf(x, y, np.absolute(fft.fftshift(fft.fft2(psi))))
#     axes[1, 0].set_title("k Space Density")
#     axes[1, 0].set_aspect('equal')
#     plt.colorbar(p1, ax=axes[1, 0])
#
#     N = solver.psiFinal.shape[0]
#     axes[0, 1].plot(x[N/2, :], np.absolute(solver.psiFinal[N/2, :]))
#     axes[0, 1].set_title("Radial Profile")
#     axes[0, 1].set_ylabel("Density")
#     axes[0, 1].set_xlabel("x")
#
#     p2 = axes[1, 1].contourf(x, y, np.angle(psi))
#     axes[1, 1].set_title("Phase")
#     axes[1, 1].set_aspect('equal')
#     plt.colorbar(p2, ax=axes[1, 1])
#
#     axes[0, 2].plot(solver.energyTimes, solver.energy)
#     axes[0, 2].set_title("Energy")
#     axes[0, 2].set_xlabel("Time")
#     axes[0, 2].set_ylabel("Energy")
#
#     p4 = axes[1, 2].contourf(x, y, solver.nFinal)
#     axes[1, 2].set_title("Resovoir density")
#     axes[1, 2].set_aspect('equal')
#     plt.colorbar(p4, ax=axes[1, 2])
#
#     axes[0, 3].plot(x[N/2, :],
#                     pumpFunction(x[N/2, :], y[N/2, :]) / solver.Pth,
#                     x[N/2, :],
#                     solver.damping[N/2, :])
#     axes[0, 3].set_title("Pump and damping")
#
#     p4 = axes[1, 3].contourf(x, y, pumpFunction(x, y) / solver.Pth)
#     axes[1, 3].set_title("Pump Density (units of Pth)")
#     # axes[1, 3].set_aspect('equal')
#     plt.colorbar(p4, ax=axes[1, 3])
#
#     fig.tight_layout()
#     return fig


class SpatialFunction(object):
    """
    Not to be used.
    Parent class for spatial functions such as the pump and potential.
    Spatial functions are required to provide a function that corresponds to
    the spatial function. Eg, for a pump, the function would take an x grid and
    y grid and output the pump power density function.
    Should also provide a characteristic size so that the grid extent may be
    scaled to it.

    The main use of these classes is to store as an attribute the scaled and
    unscaled functions, and to be able to define such functions in the scaled or
    unscaled version. This should make switching between scaled and unscaled
    versions much easier.

    Must provide at least these two methods:
        unscaledFunction: The function in SI units. The arguments are x and y
        coordinates in SI distance units. The function output is (almost always)
        in SI units (apart from pump power, which is in units of the threshold
        pump power).

        scaledFunction: The function scaled to the units of the normalised GPE.
        The arguments are the x and y coordinates in units of the characteristic
        length.
    Often also provides:
        charSize: A characteristic size associated with the function. This may
        be used when initialising a grid to ensure that the grid is large enough
        to capture the relevant region.
    """
    # TODO: Write the above more clearly (maybe still?)

    def __init__(self):
        raise NotImplementedError()

    def unscaledFunction(self):
        raise NotImplementedError()

    def scaledFunction(self, charL, charOut=1):
        return scale(self.unscaledFunction(), charL, charOut=charOut)


class GaussianPump(SpatialFunction):
    """
    A Gaussian pump

    Useful attributes:
        charSize: The FWHM radius of the spot.
    """
    # TODO: Allow for init to accept scaled and unscaled arguments.
    # TODO: Allow for ellispoidal spot shape
    # TODO: Write scaled and unscaled version
    def __init__(self, sigma, P0, Pth, exponent=1):
        """
        Make a Gaussian pump with maximum power P0 * Pth, where Pth is the
        threshold value of P in the normalised GPE.

        Parameters:
            sigma: value of sigma for the Gaussian. Should be in SI units
            P0: Maximum pump power. In units of Pth.
            Pth: The value of the threshold power in the scaled GPE.
            exponent: value of exponent in the Gaussian. Gaussian function is
            exp(-0.5 * sigma^2 * r^exp)
        """
        self.charSize = 2.0 * np.sqrt(2 * np.log(2)) * sigma
        self.P0 = P0
        self.sigma = sigma
        self.Pth = Pth
        self.exp = exponent

    def unscaledFunction(self):
        def out(x, y): return (self.P0
                               * self.Pth
                               * np.exp(-0.5*(x**2 + y**2)**self.exp
                                        / self.sigma**2))
        return out

    # def scaledFunction(self, charLength):
    #     def out(x, y): return (self.P0
    #                            * self.Pth
    #                            * np.exp(-0.5
    #                                     * ((charLength * x)**2
    #                                        + (charLength * y)**2)
    #                                     / self.sigma**2))
    #     return out


class RectangularTanhDamping(SpatialFunction):
    """
    tanh damping
    """
    def __init__(self, max_XY, decayDistance=0.15, k=3.0):
        """
        Parameters:
            max_XY: Defines the size of the grid. The grid runs from -max_XY to
            max_XY in each dimension.
            decayDistance: The damping will decay from 1 to 0 roughly between
            (1 - decayDistance)*max_XY and max_XY at each boundary.
            k: Defines how close to 0 the damping will be at the boundaries. We
            will have damping(max_XY) = tanh(-k) + 1, and
            damping((1-r) * max_XY) = tanh(k). Bigger k means a sharper decay.
        """
        self.max_XY = max_XY
        self.r = decayDistance
        self.k = k

    def unscaledFunction(self):
        a = 2*self.k / (self.r*self.max_XY)
        b = self.k * (2.0/self.r - 1)

        def out(x, y): return 0.25 * ((np.tanh(a*x + b) + np.tanh(-a*x + b)) *
                                      (np.tanh(a*y + b) + np.tanh(-a*y + b)))
        return out

    # def scaledFunction(self, charLength):
    #     max_XY_scaled = self.max_XY / charLength
    #     a = 2*self.k / (self.r*max_XY_scaled)
    #     b = self.k * (2.0/self.r - 1)

    #     def out(x, y): return 0.25 * ((np.tanh(a*x + b) + np.tanh(-a*x + b)) *
    #                                   (np.tanh(a*y + b) + np.tanh(-a*y + b)))
    #     return out

# TODO: Documentation


class RadialTanhDamping(SpatialFunction):
    """
    tanh damping
    """
    def __init__(self, max_XY, decayDistance=0.15, k=4.0):
        """
        Parameters:
            max_XY: Defines the size of the grid. The grid runs from -max_XY to
            max_XY in each dimension.
            decayDistance: The damping will decay from 1 to 0 roughly between
            (1 - decayDistance)*max_XY and max_XY at each boundary.
            k: Defines how close to 0 the damping will be at the boundaries. We
            will have damping(max_XY) = tanh(-k) + 1, and
            damping((1-r) * max_XY) = tanh(k). Bigger k means a sharper decay.
        """
        self.max_XY = max_XY
        self.r = decayDistance
        self.k = k

    def unscaledFunction(self):
        a = 2*self.k / (self.r*self.max_XY)
        b = self.k * (2.0/self.r - 1)

        def out(x, y): return 0.5 * (np.tanh(a*np.sqrt(x**2 + y**2) + b)
                                     + np.tanh(-a*np.sqrt(x**2 + y**2) + b))
        return out

    # def scaledFunction(self, charLength):
    #     max_XY_scaled = self.max_XY / charLength
    #     a = 2*self.k / (self.r*max_XY_scaled)
    #     b = self.k * (2.0/self.r - 1)

    #     def out(x, y): return 0.25 * ((np.tanh(a*x + b) + np.tanh(-a*x + b)) *
    #                                   (np.tanh(a*y + b) + np.tanh(-a*y + b)))
    #     return out


class ParabolicPump():
    """
    Parabolic pump .
    """
    def __init__(self, p, w, k, vertex=(0, 0), reverse=False, sigma=3):
        """
        Make a parabolic pump defined by p, k, and w, with vertex of the
        condenstate-side parabola at vertex. reverse controls whether the
        parabola is concave up or concave down. If reverse is True, the parabola
        is concave down. See diagram for definition of the parameters.
        sigma: value of sigma to use for the gaussian filter
        """
        self.p = p
        self.k = k
        self.w = w
        self.h, self.l = vertex
        self.reverse = reverse
        self.sigma = sigma
        self.a = 1 / (4*self.p)
        self.b = -self.h / (2*self.p)
        sign = 1.0 * (not self.reverse) - 1.0 * self.reverse
        self.sign = sign
        self.c_i = self.h**2 / (4*self.p) + self.l
        self.c_o = self.c_i - self.k

        # Functions of the parabolas. inFacing is the one on the condensate
        # side, outFacing is the other one.
        def inFacing(x): return sign * (self.a * x**2 + self.b * x + self.c_i)

        def outFacing(x): return sign * (self.a * x**2 + self.b * x + self.c_o)

        self.inFacing = inFacing
        self.outFacing = outFacing
        self.y_maxmin = self.inFacing(self.w + self.h)
        # TODO: Is this actually a good way to define the characteristic size?
        self.charSize = max(self.w, np.abs(self.y_maxmin))
    # TODO: Return smoothed functions
    # These functions should only be called on arrays, otherwise the smoothing
    # won't work.
    # TODO: We might run into trouble, because sigma for the smoothing is in
    # pixels.

    def unscaledFunction(self):
        if not self.reverse:
            def out(x, y): return 1.0 * ((y <= self.y_maxmin)
                                         & (y <= self.inFacing(x))
                                         & (y >= self.outFacing(x)))
        else:
            def out(x, y): return 1.0 * ((y >= self.y_maxmin)
                                         & (y >= self.inFacing(x))
                                         & (y <= self.outFacing(x)))
            return lambda x, y: gaussian_filter(out(x, y),
                                                self.sigma, mode='constant',
                                                cval=0.0)


class DoubleParabolicPump():
    # TODO
    pass


class AnnularPumpGaussian(SpatialFunction):
    """

    """
    def __init__(self, rMiddle, P0, Pth, sigma=3):
        self.rMiddle = rMiddle
        self.sigma = sigma
        self.P0 = P0 * Pth

    def unscaledFunction(self):
        """

        """
        def out(x, y):
            return self.P0 * np.exp(- 0.5 * (x**2 + y**2
                                             - self.rMiddle**2) ** 2
                                    / self.sigma**4)
        return out


class AnnularPumpFlat(SpatialFunction):
    """
    Fixed so that you are garuanteed to get a pump with maximum power P0 * Pth.

    Note sigma is in units of pixels. You should use grid.toPixels to convert
    the diffusion length of excitons to pixels. Otherwise the pump function will
    change as you change the resolution and spatial extent of the grid!
    """
    def __init__(self, rMiddle, width, P0, Pth, sigma):
        self.rMiddle = rMiddle
        self.width = width
        self.P0 = P0 * Pth
        self.sigma = sigma

    def unscaledFunction(self):
        def out(x, y):
            mask = ((x**2 + y**2 >= (self.rMiddle - self.width)**2)
                    & (x**2 + y**2 <= (self.rMiddle + self.width)**2)).\
                astype(int)
            mask *= 1000
            mask = gaussian_filter(mask, self.sigma, mode='constant',
                                   cval=0.0)
            # Rescale the mask so that the max value is P0
            mask = self.P0 * mask / np.max(mask)
            return mask
        return out

        # def smooth(x, y):
        #     return gaussian_filter(mask(x, y), self.sigma, mode='constant',
        #                            cval=0.0)
        # return lambda x, y: gaussian_filter(mask(x, y), self.sigma,
        #                                     mode='constant', cval=0.0)
        # def out(x, y):
        #     smoothed = smooth(x, y)
        #     # Normalise
        #     maxVal = np.max(smoothed)
        #     return self.P0 * smoothed / maxVal
        # return out
