"""
Module that contains classes used to model the coupled GPE using split-step
fourier method
"""
from __future__ import division
from math import floor
import numpy as np
from numpy import fft
# import pyfftw
import UtilityFunctions
import os
import h5py
import pint
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import pycuda.driver as drv

# TODO: Or use a better units package

WISDOM_LOCATION = os.path.join(os.path.expanduser('~'), '.wisdom', 'wisdom')


class ParameterContainer(object):
    """
    This class serves as a container for the values of the parameters used in an
    experiment, and provides access to the scaling factors.
    """
    __requiredArgs = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'm']
    __doubleCompArgs = []
    __singleCompArgs = ['gamma_nl']
    __allowedArgs = ['R', 'g_C', 'g_R', 'gamma_R', 'gamma_C', 'gamma_eff',
                     'gamma_nl', 'm', 'charT', 'charL', 'k']
    __outArgsSingle = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'Pth', 'charL',
                       'charT']
    __outArgsDouble = __outArgsSingle + ["gamma_nl"]

    def __init__(self, **kwargs):
        """
        Initialises all the parameters and computes the scaling factors.
        All parameters should be pint Quantities.

        :Parameters:
            **kwargs:
                The following are all required:
                    R : Stimulated scattering rate
                    g_C : Polariton-polariton interaction strength
                    g_R:
                    gamma_C : Polariton relaxation rate
                    gamma_R
                    m : Polariton effective mass, in units of the electron mass
                The following are optional:
                    k: Value of dispersive coefficient. If not provided,
                    defaults to hbar**2 / (2 * m)
                    charT: a characteristic time to scale by. If not provded,
                    defaults to (gamma_C) ^ -1
                    charL: a charactersitic length to scale by. If not provided,
                    defaults to ( hbar * charT / (2 * m * gamma_C) )^1/2.
        """
        self.singleComp = True if 'gamma_nl' in kwargs else False
        self.__checkArgs(kwargs)

        # Define the output units and actual units of the parameters
        # We'll make them strings to make life easy, and because we want them to
        # be reported int the right way
        self.ureg = pint.UnitRegistry()
        m_e = self.ureg.electron_mass.to_base_units().magnitude
        hbar = self.ureg.hbar.to_base_units().magnitude
        self.gOutput = "millieV * micrometer**2"
        self.rOutput = "millieV * micrometer**2 * hbar**-1"
        self.gammaOutput = "picosecond ** -1"
        self.gammaNlBase = "meter**2 * second**-1"
        self.mOutput = "electron_mass"
        self.charLOutput = "micrometer"
        self.charTOutput = "picosecond"
        self.gammaNlOutput = "millieV * micrometer**2 * hbar**-1"
        self.mBase = "gram"
        self.gBase = "gram * meter **4 * second ** -2"
        self.rBase = "meter**2 * second **-1"
        self.gammaBase = "second ** -1"
        self.charLBase = "meter"
        self.charTBase = "second"

        # Read in the keyword arguments
        for (k, v) in kwargs.items():
            setattr(self, k, v.to_base_units().magnitude)

        # Set mass. We read in mass in units of electron mass for convenience,
        # but it must be converted to SI units
        # self.m_scaled = self.m
        # self.m = self.__class__.__m_e * self.m

        # m is now read in as a pint quantity. We don't need to scale it up by
        # the electron mass, but we do need to find the scaled mass
        self.m_scaled = self.m / m_e
        # Read in k or set to default
        self.k = kwargs.get('k', hbar**2 / (2 * self.m))

        # Define our characteristic length, time, and energy scales.
        # If t' is the (nondimensional) time variable used in the
        # nondimensionalised GPE, then t = charT * t'. For example, R' is the
        # stimulated scattering rate used in the normalised GPE, so R = charR *
        # R'. If they are not provided, we will set them to the default, which
        # is the scaling that makes k=1 and gamma'_C = 1.

        self.charT = kwargs.get('charT', 1.0 / self.gamma_C)
        if 'charT' in kwargs.keys():
            self.charT = self.charT.to_base_units().magnitude
        self.charL = kwargs.get('charL', np.sqrt((hbar
                                                  * self.charT)
                                                 / (2.0 * self.m)))
        if 'charL' in kwargs.keys():
            self.charL = self.charL.to_base_units().magnitude
        # A characteristic energy
        self.charU = hbar / self.charT
        self.charg = (hbar * self.charL**2) / self.charT
        self.charR = self.charL ** 2 / self.charT
        self.charGamma = 1.0 / self.charT
        self.charGammaNl = self.charL ** 2 / self.charT
        # TODO: Check
        self.chark = (hbar * self.charL**2) / self.charT
        # This may not be required - the P term in the GPE is phenomonological,
        # and the experimentalist probably only knows it in terms of Pth
        self.charP = 1.0 / (self.charT * self.charL ** 2)

        # Scaled parameters - these are the ones to used in the
        # nondimensionalised GPE
        self.g_C_scaled = self.g_C / self.charg
        self.gamma_C_scaled = self.gamma_C / self.charGamma
        self.k_scaled = self.k / self.chark
        self.g_R_scaled = self.g_R / self.charg
        self.gamma_R_scaled = self.gamma_R / self.charGamma
        self.R_scaled = self.R / self.charR
        # Compute threshold pump power for the normalised GPE.
        self.Pth_scaled = ((self.gamma_R_scaled * self.gamma_C_scaled)
                           / self.R_scaled)
        # Compute threshold pump power for unnormalised GPE. We can get this
        # from the scaled one.
        self.Pth = self.charP * self.Pth_scaled

        if self.singleComp:
            self.gamma_nl_scaled = self.gamma_nl / self.charGammaNl

    def __checkArgs(self, kwargs):
        """
        Checks the consistency and validity of the keyword arguments. Raises
        errors if the system is not completely specified, or if unrecognised
        arguments are provided.
        """
        requiredArgs = self.__class__.__requiredArgs + \
            self.__class__.__singleCompArgs if self.singleComp else\
            self.__class__.__requiredArgs + self.__class__.__doubleCompArgs
        for arg in requiredArgs:
            if arg not in kwargs:
                raise ValueError("Essential keyword argument %s missing" % arg)
        for (k, v) in kwargs.items():
            assert k in self.__class__.__allowedArgs, "Invalid Argument %s" % k

    def getGPEParams(self):
        """
        Returns a dictionary containing the parameters in the nondimensionalised
        GPE, and a few other useful parameters, such as Pth. Mostly used for
        calling GPESolver.

        Returns:
            A dictionary with the following keys and values for the
            nondimensionalised GPE:
                R : Stimulated scattering rate.
                gamma_C : Polariton decay rate.
                gamma_R : Exciton decay rate.
                g_C : Polariton-polariton interaction strength.
                g_R : Polariton-exciton interaction strength.
                Pth : Threshold pump power.
        """
        outKeysScaleDouble = ['R', 'gamma_C', 'gamma_R', 'g_C', 'g_R', 'k',
                              'Pth']
        outKeysScaleSingle = outKeysScaleDouble + ['gamma_nl']
        outKeysScale = outKeysScaleSingle if self.singleComp else\
            outKeysScaleDouble
        outKeys = ['charL', 'charT']
        out = {key: self.__dict__[key + '_scaled'] for key in outKeysScale}
        for key in outKeys:
            out[key] = self.__dict__[key]
        return out

    def getOutputParams(self):
        """
        Returns a dictionary with the GPE parameters in useful units. Doesn't
        include g_R because it's always 2 * g_C
        """
        out = {}
        out["g_C"] = self.ureg.Quantity(self.g_C, self.gBase).to(self.gOutput)
        out["gamma_C"] = self.ureg.Quantity(self.gamma_C,
                                            self.gammaBase).to(self.gammaOutput)
        out["m"] = self.ureg.Quantity(self.m, self.mBase).to(self.mOutput)
        out["charL"] = self.ureg.Quantity(self.charL,
                                          self.charLBase).to(self.charLOutput)
        out["charT"] = self.ureg.Quantity(self.charT,
                                          self.charTBase).to(self.charTOutput)
        out["g_R"] = self.ureg.Quantity(self.g_R, self.gBase).to(self.gOutput)
        out["R"] = self.ureg.Quantity(self.R, self.rBase).to(self.rOutput)
        out["gamma_R"] = self.ureg.Quantity(self.gamma_R,
                                            self.gammaBase).to(self.gammaOutput)
        if self.singleComp:
            out["gamma_nl"] = self.ureg.Quantity(self.gamma_nl,
                                                 self.gammaNlBase).\
                to(self.gammaNlOutput)
        return out


class Grid(object):
    """
    A class that contains derived GPE parameters and uses them to create scaled
    and unscaled grids.
    """
    # TODO: Allow us to provide a characteristic size in scaled units.
    def __init__(self, charLength, max_XY, N=1024):
        """
        Initialise a grid, either of a particular size, or of a size large
        enough to contain some object which is defined by a characteristic size.
        We define grids internally only through one-dimensional arrays that
        represent the axes. Spatials grids are returned by the getFooGrid
        methods.

        Parameters:
            charLength: The characteristic length that will be used to obtain
            the scaled grid. Typically obtained from a ParameterContainer
            instance

            N: Number of grid elements along each axis. The grid will be N * N.
            Defaults to 1024

            max_XY: the maximum spatial extent of the grid, a pint Quantity. If
            provided, the grid will run from -max_XY to max_XY in both
            dimensions.
        """
        # Check that N is a power of 2
        if bin(N).rfind('1') != 2:
            raise Warning("N is not a power of 2. FFTs may be slow!")
        self.max_XY_unscaled = max_XY.to_base_units().magnitude
        self.charL = charLength
        self.N = N
        self.max_XY_scaled = self.max_XY_unscaled / self.charL
        # Unscaled grid. x and y values are in SI units.
        self.x_axis_unscaled = np.linspace(- self.max_XY_unscaled,
                                           self.max_XY_unscaled, num=N)
        self.dx_unscaled = np.absolute(self.x_axis_unscaled[0]
                                       - self.x_axis_unscaled[1])
        self.dx_scaled = self.dx_unscaled / self.charL
        # These must be fftshift'ed before making a grid
        self.k_axis_unscaled = (((2.0 * np.pi) / (2.0 * self.max_XY_unscaled)) *
                                np.linspace(-N/2, N/2 - 1, num=N))
        # Grid in units of the characteristic length provided in params
        self.x_axis_scaled = np.linspace(
            -self.max_XY_scaled, self.max_XY_scaled, num=N)
        self.k_axis_scaled = (((2.0 * np.pi) / (2.0 * self.max_XY_scaled)) *
                              np.linspace(-N/2, N/2 - 1, num=N))

    def getSpatialGrid(self, scaled=True):
        """
        Build and return a spatial grid.

        parameters:
            scaled: If False, the values of the grid will correspond to SI units
            of length. If True, the values will be in units of self.charLength.
        returns:
            x, y: Arrays of x and y values corresponding to the grid
        """
        if scaled:
            return np.meshgrid(self.x_axis_scaled, self.x_axis_scaled)
        else:
            return np.meshgrid(self.x_axis_unscaled, self.x_axis_unscaled)

    def getkGrid(self, scaled=True):
        """
        Build and return a grid of k values.

        parameters:
            scaled: If False, the values of the grid will correspond to SI units
            of 1 / length. If True, the values will be in units
            of 1 / self.charLength.
        Returns:
            k_x, k_y: Arrays of k_x and k_y values corresponding to the grid
        """
        if scaled:
            return np.meshgrid(fft.fftshift(self.k_axis_scaled),
                               fft.fftshift(self.k_axis_scaled))
        else:
            return np.meshgrid(fft.fftshift(self.k_axis_unscaled),
                               fft.fftshift(self.k_axis_scaled))

    def getKSquaredGrid(self, scaled=True):
        """
        Build and return a grid of k_x^2 + k_y^2 values.

        Parameters:
           scaled: If False, the values of the grid will correspond to SI units
           of 1 / length. If True, the values will be in units
           of 1 / self.charLength.

        Returns:
            An array corresponding to the function k -> k_x^2 + k_y^2
        """
        kx, ky = self.getkGrid(scaled=scaled)
        return kx ** 2 + ky ** 2

    def toPixels(self, l, scaled=False):
        """
        Converts a length to pixels:

        Parameters:
            l: Length to convert. If the scaled is true, the length is in units
            of charL. If not, it is in base units (the same as charL).
            scaled: whether the length is in units of charL or not.

        Returns:
            The length in pixels (not rounded).
        """
        if scaled:
            lScaled = l

        else:
            lScaled = l / self.charL
        return lScaled / self.dx_scaled


class GPESolver(object):
    """
    Class that solves an instance of the GPE and keeps some results.

    The GPESolver object simply stores the grid and FFTW objects
    that perform the FFTs. Its solve method
    may be called many times with different pumps, potentials and initial
    wavefunctions. This should result in a slight speed increase when many
    experiments are performed as we can make fftw plans outside of the solve
    method.

    Calling the solve method causes the final condensate wavefunction and
    resovoir density to be stored as attributes of the GPESolver.
    """
    # Strictly speaking, Pth is not required, but we will pass the output of
    # ParameterContainer.getGPEParams, so it will be present.
    __requiredParams = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'Pth', 'k',
                        'charL', 'charT']
    __stepsPerTimePrint = 1e3

    def __init__(self, paramContainer, dt, grid, pumpFunction=None,
                 damping='default', psiInitial=None, REAL_DTYPE=np.float32,
                 COMPLEX_DTYPE=np.complex64):
        """
        Initialise an instance of a GPESolver.

        Parameters:

            spatialGrid: A tuple (x, y) representing the spatial grid that the
            simulation will be performed on. This grid should be scaled to units
            of the characteristic length defined in ParameterContainer.

            kGrid: A tuple (k_x, k_y) representing the k-space grid
            corresponding to the (x, y) grid. The scaling of this grid should
            correspond to that of the (x, y) grid. That is, it should be scaled
            to units of the inverse of the characterestic length defined in
            ParameterContainer.

            damping: The damping method to use in order to suppress the implicit
            periodic boundary conditions. Default is a tanh function that
            drops from 1.0 to 0 over the last 10% of each dimension. Presently,
            the only other option is "None", which means no damping, and hence
            periodic boundary conditions


            FFTW_METHOD: The method for FFTW to use when planning the transforms
            FFTW_PATIENT, FFTW_EXHAUSTIVE and FFTW_MEASURE will result in
            faster transforms, but may take a significant amount of time to plan

            N_THREADS: The number of threads for FFTW to use. Currently 2
            threads seems to give the lowest time per step (8 core computer).
            Increasing this may greatly increase the time that FFTW requires to
            plan the transforms.
        """
        self.grid = grid
        self.paramContainer = paramContainer
        params = paramContainer.getGPEParams()
        self.singleComp = True if 'gamma_nl' in params else False
        self.x, self.y = grid.getSpatialGrid()
        self.dx_scaled = grid.dx_scaled
        self.dx_unscaled = grid.dx_unscaled
        self.K = grid.getKSquaredGrid()
        # This is already scaled because we obtained it from the scaled grid.
        self.max_XY = np.abs(self.x[-1, -1])
        self.N = int(self.x.shape[0])
        self.dt = dt
        self.time = 0
        self.nSteps = 0
        assert self.x.shape == self.y.shape, "Spatial grids are not the same\
               shape"
        assert self.x.shape == self.K.shape, "Spatial grids are not the same\
               shape as k grid"
        assert self.x.shape == (self.N, self.N), 'Grid must be square.'

        # Check we have the required parameters and assign them
        for key in self.__class__.__requiredParams:
            if key not in params:
                raise ValueError("Required Parameter %s missing" % key)
            self.__setattr__(key, params[key])
        if self.singleComp:
            self.gamma_nl = params["gamma_nl"]
        self.max_XY_scaled = self.max_XY * self.charL
        # Create arrays for damping, etc.
        if damping == 'default':
            damping = UtilityFunctions.RadialTanhDamping(
                self.max_XY, k=10.0).unscaledFunction()(self.x, self.y)
        elif damping == 'rectangular':
            damping = UtilityFunctions.RectangularTanhDamping(
                self.max_XY, k=10).unscaledFunction()(self.x, self.y)
        self.damping = damping
        if not psiInitial:
            psi0 = (np.abs(np.random.normal(size=(self.N, self.N),
                                            scale=10e-5, loc=10e-4))
                    + 0.0j*np.random.normal(size=(self.N, self.N),
                                            scale=10e-4, loc=10e-3))
        else:
            psi0 = psiInitial(self.x, self.y) + 0j
        currentDensity = np.absolute(psi0) ** 2
        if not pumpFunction:
            pumpFunction = lambda x, y: np.zeros_like(x)
        Pdt = pumpFunction(self.x, self.y) * self.dt
        expFactorPolFirst = self.dt * (0.5 * self.R - 1j * self.g_R)
        expFactorPolThird = -0.5 * self.gamma_C * self.dt

        if self.singleComp:
            expFactorPolSecond = - (self.gamma_nl + 1j * self.g_C) * self.dt
            n = Pdt / (self.dt * self.gamma_R)
        else:
            expFactorPolSecond = - 1j * self.g_C * self.dt
            n = np.zeros_like(psi0, dtype=np.float64)
        kineticFactorHalf = np.exp(-1.0j * self.k * self.K * self.dt / 2.0)

        from skcuda.misc import add, multiply
        from pycuda.cumath import exp
        self.add = add
        self.multiply = multiply
        self.exp = exp
        self.cu_fft = cu_fft

        self.damping = gpuarray.to_gpu(damping.astype(REAL_DTYPE))
        self.psi = gpuarray.to_gpu(psi0.astype(COMPLEX_DTYPE))
        self.n = gpuarray.to_gpu(n.astype(REAL_DTYPE))
        self.currentDensity = gpuarray.to_gpu(currentDensity
                                              .astype(REAL_DTYPE))
        self.Pdt = gpuarray.to_gpu(Pdt.astype(REAL_DTYPE))
        self.expFactorPolFirst = gpuarray.to_gpu(
            np.array([expFactorPolFirst]).astype(COMPLEX_DTYPE))
        self.expFactorPolSecond = gpuarray.to_gpu(
            np.array([expFactorPolSecond]).astype(COMPLEX_DTYPE))
        self.expFactorPolThird = gpuarray.to_gpu(
            np.array([expFactorPolThird]).astype(COMPLEX_DTYPE))
        self.kineticFactorHalf = gpuarray.to_gpu(
            np.array([kineticFactorHalf]).astype(COMPLEX_DTYPE))
        # self.expFactorExciton = gpuarray.to_gpu(expFactorExciton)
        self.plan_forward = cu_fft.Plan((self.N, self.N), COMPLEX_DTYPE,
                                        COMPLEX_DTYPE)
        self.plan_inverse = cu_fft.Plan((self.N, self.N), COMPLEX_DTYPE,
                                        COMPLEX_DTYPE)

    def step(self, printTime=False):
        """
        Make a single timestep.
        """
        self.step_gpu()
        self.nSteps += 1
        self.time += self.dt
        if printTime and\
                (self.nSteps % self.__class__.__stepsPerTimePrint == 0):
            print("Time = %f" % self.time)

    def step_gpu(self):
        self.cu_fft.fft(self.psi, self.psi, self.plan_forward)
        self.psi *= self.kineticFactorHalf
        self.cu_fft.ifft(self.psi, self.psi, self.plan_inverse, scale=True)

        # currentDensity_gpu = abs(psi_gpu) ** 2
        self.currentDensity = (self.psi * self.psi.conj()).real

        # expFactorExciton_gpu = cumath.exp(-gammaRdt_gpu +
        #                                   (Rdt_gpu * currentDensity_gpu))

        # # TODO: Don't bother creating the intermediate? Write this as a single
        # # pycuda elementwise kernel
        # n_gpu *= cumath.exp(-gammaRdt_gpu + Rdt_gpu * currentDensity_gpu)
        if not self.singleComp:
            self.n *= self.exp(self.add(- self.gammaRdt,
                                        - self.multiply(self.Rdt,
                                                        self.currentDensity)))
            self.n += self.Pdt

        self.psi *= self.exp(
            self.add(
                self.add(self.multiply(self.expFactorPolFirst, self.n),
                         self.multiply(self.expFactorPolSecond,
                                       self.currentDensity)),
                self.expFactorPolThird))

        # self.psi *= self.exp(self.expFactorPolFirst * self.n
        #                      + self.expFactorPolSecond*self.currentDensity
        #                      + self.expFactorPolThird)

        self.cu_fft.fft(self.psi, self.psi, self.plan_forward)
        self.psi *= self.kineticFactorHalf
        self.cu_fft.ifft(self.psi, self.psi, self.plan_inverse, scale=True)

        self.psi *= self.damping
        if not self.singleComp:
            self.n *= self.damping

    def stepTo(self, t_max, recordEnergy=True, normalized=True,
               recordNumber=True, stepsPerObservation=100, printTime=False,
               recordSpectrum=True, collectionMask=None, spectStart=50,
               spectLength=50, spectMax=None):
        """
        Step until t is t_max.

        Parameters:
            t_max: Time to step to. When stepping is finished, time will be
            within dt of t_max
            recordEnergy: Flag to turn on energy recording.
            recordNumber: Flag to turn on recording of total number.
            stepsPerObservation: Number of steps between each energy observation
            energyMask: Mask array to collect energy data from a subset of the
            integration region.
            spectMax: index used to collect spectral data from a subset of the
            integration region. If not None, slices will be collected from
            the slice [N/2 - spectMax, N/2 + spectMax]. Defaults to N/2 so that
            by default the entire integration region is collected.
        Returns:
                None
        """
        # Spectrums can only be recorded along a horizontal slice. Defaults to
        # middle. It's not clear if it will be ahrd to record along vertical.
        N_TIMESTEPS = int(floor((t_max - self.time) // self.dt))
        # Initialize arrays for observation times and observations.
        if recordEnergy:
            self.energy = np.zeros(int(floor((N_TIMESTEPS
                                              // stepsPerObservation))) + 1)
            self.times = np.zeros_like(self.energy)
            self.numObservations = 0
        if recordNumber:
            self.number = np.zeros(int(floor((N_TIMESTEPS
                                              // stepsPerObservation))) + 1)
            self.times = np.zeros_like(self.number)
            self.numObservations = 0
        if recordSpectrum:
            if spectStart + spectLength > t_max:
                raise ValueError("Spectrum parameters not compatible\
                                 with t_max")
            if spectStart < self.time:
                raise ValueError("Spectrum start time is before current time")
            spectMax = self.N if not spectMax else spectMax
            self.numSpectObs = spectLength // self.dt
            # StartStep and EndStep are the first and last steps during which a
            # spectrum slice will be recorded.
            self.spectStartStep = spectStart // self.dt
            self.spectEndStep = self.spectStartStep + self.numSpectObs - 1
            # This array stores slices (along the second axis) for different
            # times (along the first axis)
            self.spectrum_gpu = gpuarray.to_gpu(np.zeros((self.numSpectObs,
                                                          2*spectMax),
                                                         dtype=self.psi.dtype))
            self.omega_axis = np.linspace(-self.numSpectObs / 2.0,
                                          self.numSpectObs / 2.0 - 1,
                                          num=self.numSpectObs) *\
                2*np.pi / (self.numSpectObs * self.dt)
            spectStep = 0

        for step in xrange(N_TIMESTEPS):
            self.step(printTime=printTime)
            if recordEnergy and step % stepsPerObservation == 0:
                self.energy[self.numObservations] = \
                    self.getEnergy(normalized=normalized, mask=collectionMask)
            if recordNumber and step % stepsPerObservation == 0:
                self.number[self.numObservations] = \
                    self.getTotalNumber(mask=collectionMask)
            if (recordEnergy or recordNumber) and\
                         step % stepsPerObservation == 0:
                self.times[self.numObservations] = self.time
                self.numObservations += 1

            # if (recordEnergy or recordNumber) and\
            #         step % stepsPerObservation == 0:
            #     self.times[self.numObservations] = self.time
            #     if recordEnergy:
            #         self.energy[self.numObservations] = \
            #             self.getEnergy(normalized=normalized,
            #                            mask=collectionMask)
            #     if recordNumber:
            #         self.number[self.numObservations] =\
            #             self.getTotalNumber(mask=collectionMask)
            #     self.numObservations += 1
            if recordSpectrum and step >= self.spectStartStep\
                    and step <= self.spectEndStep:
                # Note that we need to copy into the spectrum array in reverse
                # order. Ie, fill from the bottom up. This is because the omega
                # axis of the spectrum is reversed.
                drv.memcpy_dtod(self.spectrum_gpu[- spectStep, :].gpudata,
                                self.psi[self.N//2, self.N//2 - spectMax:
                                         self.N//2 + spectMax].gpudata,
                                self.psi[self.N//2, self.N//2 - spectMax:
                                         self.N//2 + spectMax].nbytes)
                spectStep += 1

        # change the gpu spectrum array into a useable one. We can just do a 2d
        # fft since we recorded the wavefunction rather than the momentum
        if recordSpectrum:
            self.spectrum = fft.fftshift(fft.fft2(self.spectrum_gpu.get()))

        if recordEnergy or recordNumber:
            out = [self.times]
            if recordEnergy:
                out.append(self.energy)
            if recordNumber:
                out.append(self.number)
            return out

    def getEnergy(self, normalized=True, mask=None):
        """
        Returns the current energy.

        Parameters:
            normalized: Flag to return the normalized energy (that is, divided
            by the total density)
        """
        psi = self.psi.get()
        n = self.n.get()
        density = np.absolute(psi) ** 2
        gradx = np.gradient(psi)[0]
        normFactor = density.sum() if normalized else 1.0
        return np.ma.array(-(0.25 * np.gradient(
            np.gradient(density)[0])[0]
            - 0.5 * np.absolute(gradx) ** 2
            - (self.g_C * density + self.g_R * n)
            * density), mask=mask).sum() / normFactor

    def getTotalNumber(self, mask=None):
        """
        Returns the total number of particles. Performs appropriate scaling on
        the wavefunction.

        Will be incorrect if a mask is used as it calculates the density and
        multiplies by the area, and the area of the mask is not taken into
        account.
        """
        densityScaled = self.getDensity(scaled=True, mask=None)
        area = (2 * self.max_XY_scaled) ** 2
        return densityScaled.sum() * area

    def getDensity(self, scaled=False, mask=None):
        """
        Returns the density, performing scaling if appropriate.

        Parameters:
            scaled: Flag to scale the density.
            mask: Collection mask. Density will be incorrect if a mask is used,
            becuase the area will be incorrect. You can fix this yourself?!

        Returns:
            The density distribution. If scaled is False, the density will be in
            units of number / (charL) ** 2. If scaled is True, the density
            will be in units of number / (micron) ** 2.
        """
        psi = self.psi.get()
        if mask:
            psi = np.ma.array(psi, mask=mask)
            return np.absolute(psi) * self.charL**(-2) \
                if scaled else np.absolute(psi)**2
        else:
            return np.absolute(psi) * self.charL**(-2) \
                if scaled else np.absolute(psi)**2

    def save(self, fName, **kwargs):
        """
        Saves the relevant parts of the solver as an hdf5 file.
        The parameters are saved as attributes. They are saved as the parameters
        that are required to create a new paramatercontainer instance.

        Paramters:
            f: file name.
            kwargs: Other attributes to save.
        """
        f = h5py.File(fName + ".hdf5", "w")
        f.create_dataset("psi", data=self.psi.get().astype(np.complex64))
        f.create_dataset("n", data=self.n.get().astype(np.float32))
        f.create_dataset("x_ax",
                         data=self.grid.x_axis_scaled.astype(np.float32))
        f.create_dataset("k_ax",
                         data=self.grid.k_axis_scaled.astype(np.float32))
        f.create_dataset("Pdt", data=self.Pdt.get().astype(np.float32))
        if hasattr(self, "energy"):
            f.create_dataset("energy", data=self.energy.astype(np.float32))
        if hasattr(self, "number"):
            f.create_dataset("number", data=self.number.astype(np.float32))
        if hasattr(self, "times"):
            f.create_dataset("times", data=self.times.astype(np.float32))
        if hasattr(self, "spectrum"):
            f.create_dataset("spectrum",
                             data=self.spectrum.astype(np.complex64))
            f.create_dataset("omega_axis",
                             data=self.omega_axis.astype(np.float32))

        # paramsToSave = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'm', 'charT',
        #                 'charL']
        for (param, value) in self.paramContainer.getOutputParams().items():
            f.attrs[param] = value.magnitude
        for (attr, value) in kwargs.items():
            f.attrs[attr] = value
        f.attrs["t"] = self.time
        f.close()

# TODO: Provide a potential function?


class GPESolverSingle(GPESolver):
    def __init__(self, paramContainer, dt, grid, potentialFunction=None,
                 damping='default', psiInitial=None, gpu=True,
                 REAL_DTYPE=np.float32, COMPLEX_DTYPE=np.complex64):
        """
        Initialise an instance of a GPESolver.

        Parameters:

            potentialFunction: A function that defines the potential. Should
            return pint quantities.

            spatialGrid: A tuple (x, y) representing the spatial grid that the
            simulation will be performed on. This grid should be scaled to units
            of the characteristic length defined in ParameterContainer.

            damping: The damping method to use in order to suppress the implicit
            periodic boundary conditions. Default is a tanh function that
            drops from 1.0 to 0 over the last 10% of each dimension. Presently,
            the only other option is "None", which means no damping, and hence
            periodic boundary conditions


            FFTW_METHOD: The method for FFTW to use when planning the transforms
            FFTW_PATIENT, FFTW_EXHAUSTIVE and FFTW_MEASURE will result in
            faster transforms, but may take a significant amount of time to plan

            N_THREADS: The number of threads for FFTW to use. Currently 2
            threads seems to give the lowest time per step (8 core computer).
            Increasing this may greatly increase the time that FFTW requires to
            plan the transforms.
        """

        self.gpu = gpu
        self.grid = grid
        self.paramContainer = paramContainer
        params = paramContainer.getGPEParams()
        self.x, self.y = grid.getSpatialGrid()
        self.dx_scaled = grid.dx_scaled
        self.dx_unscaled = grid.dx_unscaled
        self.K = grid.getKSquaredGrid()
        # This is already scaled because we obtained it from the scaled grid.
        self.max_XY = np.abs(self.x[-1, -1])
        self.N = int(self.x.shape[0])
        self.dt = dt
        self.time = 0
        self.nSteps = 0
        assert self.x.shape == self.y.shape, "Spatial grids are not the same\
               shape"
        assert self.x.shape == self.K.shape, "Spatial grids are not the same\
               shape as k grid"
        assert self.x.shape == (self.N, self.N), 'Grid must be square.'

        # Check we have the required parameters and assign them
        # for key in self.__class__.__requiredParams:
        #     if key not in params:
        #         raise ValueError("Required Parameter %s missing" % key)
        #     self.__setattr__(key, params[key])

        # Assign parameters
        for key, value in params.iteritems():
            self.__setattr(key, value)

        self.max_XY_scaled = self.max_XY * self.charL
        # Create arrays for damping, etc.
        if damping == 'default':
            damping = UtilityFunctions.RadialTanhDamping(
                self.max_XY, k=10.0).unscaledFunction()(self.x, self.y)
        elif damping == 'rectangular':
            damping = UtilityFunctions.RectangularTanhDamping(
                self.max_XY, k=10).unscaledFunction()(self.x, self.y)
        self.damping = damping
        if not psiInitial:
            psi0 = (np.abs(np.random.normal(size=(self.N, self.N),
                                            scale=10e-5, loc=10e-4))
                    + 0.0j*np.random.normal(size=(self.N, self.N),
                                            scale=10e-4, loc=10e-3))
        else:
            psi0 = psiInitial(self.x, self.y) + 0j
        n = np.zeros_like(psi0, dtype=np.float64)
        currentDensity = np.absolute(psi0) ** 2
        if not potentialFunction:
            potentialFunction = lambda x, y: np.zeros_like(x)

        # Potential function comes with units. We need to check that these are
        # units of energy, convert them to base units, and scale to a
        # dimensionless quantity by dividing by the characteristic energy given
        # parameter container.
        Vdt = potentialFunction(self.x, self.y) * self.dt
        Vdt.ito("millieV")
        Vdt.ito_base_units()
        Vdt = Vdt.magnitude
        Vdt /= paramContainer.charU
        expFactorPolFirst = - self.dt * (1j*self.g_C + self.gamma_nl)
        expFactorPolSecond = self.gamma_eff * self.dt
        # TODO: expFactorPolThird
        kineticFactorHalf = np.exp(-1.0j * self.k * self.K * self.dt / 2.0)

        if self.gpu:
            from skcuda.misc import add, multiply
            from pycuda.cumath import exp
            self.add = add
            self.multiply = multiply
            self.exp = exp
            self.cu_fft = cu_fft

            self.damping = gpuarray.to_gpu(damping.astype(REAL_DTYPE))
            self.psi = gpuarray.to_gpu(psi0.astype(COMPLEX_DTYPE))
            self.n = gpuarray.to_gpu(n.astype(REAL_DTYPE))
            self.currentDensity = gpuarray.to_gpu(currentDensity
                                                  .astype(REAL_DTYPE))
            self.Vdt = gpuarray.to_gpu(Vdt.astype(REAL_DTYPE))
            self.expFactorPolFirst = gpuarray.to_gpu(
                np.array([expFactorPolFirst]).astype(COMPLEX_DTYPE))
            self.expFactorPolSecond = gpuarray.to_gpu(
                np.array([expFactorPolSecond]).astype(COMPLEX_DTYPE))
            self.kineticFactorHalf = gpuarray.to_gpu(
                np.array([kineticFactorHalf]).astype(COMPLEX_DTYPE))
            # self.expFactorExciton = gpuarray.to_gpu(expFactorExciton)
            self.plan_forward = cu_fft.Plan((self.N, self.N), COMPLEX_DTYPE,
                                            COMPLEX_DTYPE)
            self.plan_inverse = cu_fft.Plan((self.N, self.N), COMPLEX_DTYPE,
                                            COMPLEX_DTYPE)

    def step_gpu(self):
        self.cu_fft.fft(self.psi, self.psi, self.plan_forward)
        self.psi *= self.kineticFactorHalf
        self.cu_fft.ifft(self.psi, self.psi, self.plan_inverse, scale=True)

        # currentDensity_gpu = abs(psi_gpu) ** 2
        self.currentDensity = (self.psi * self.psi.conj()).real

        # expFactorExciton_gpu = cumath.exp(-gammaRdt_gpu +
        #                                   (Rdt_gpu * currentDensity_gpu))

        # # TODO: Don't bother creating the intermediate? Write this as a single
        # # pycuda elementwise kernel
        # n_gpu *= cumath.exp(-gammaRdt_gpu + Rdt_gpu * currentDensity_gpu)

        self.psi *= self.exp(
            self.add(
                self.add(self.multiply(self.expFactorPolFirst,
                                       self.currentDensity),
                         -1j*self.Vdt),
                self.expFactorPolSecond))

        # self.psi *= self.exp(self.expFactorPolFirst * self.n
        #                      + self.expFactorPolSecond*self.currentDensity
        #                      + self.expFactorPolThird)

        self.cu_fft.fft(self.psi, self.psi, self.plan_forward)
        self.psi *= self.kineticFactorHalf
        self.cu_fft.ifft(self.psi, self.psi, self.plan_inverse, scale=True)

        self.psi *= self.damping

    # TODO: Check
    def getEnergy(self, normalized=True, mask=None):
        """
        Returns the current energy.

        Parameters:
            normalized: Flag to return the normalized energy (that is, divided
            by the total density)
        """
        if self.gpu:
            psi = self.psi.get()
            V = self.Vdt.get() / self.dt
        else:
            psi = self.psi
            V = self.Vdt.get() / self.dt
        density = np.absolute(psi) ** 2
        gradx = np.gradient(psi)[0]
        normFactor = density.sum() if normalized else 1.0
        return np.ma.array(-(0.25 * np.gradient(
            np.gradient(density)[0])[0]
            - 0.5 * np.absolute(gradx) ** 2
            - (self.g_C * density + V)
            * density), mask=mask).sum() / normFactor

    def save(self, fName, **kwargs):
        """
        Saves the relevant parts of the solver as an hdf5 file.
        The parameters are saved as attributes. They are saved as the parameters
        that are required to create a new paramatercontainer instance.

        Paramters:
            f: file name.
            kwargs: Other attributes to save.
        """
        f = h5py.File(fName + ".hdf5", "w")
        f.create_dataset("psi", data=self.psi.get().astype(np.complex64))
        f.create_dataset("n", data=self.n.get().astype(np.float32))
        f.create_dataset("x_ax",
                         data=self.grid.x_axis_scaled.astype(np.float32))
        f.create_dataset("k_ax",
                         data=self.grid.k_axis_scaled.astype(np.float32))
        f.create_dataset("Vdt", data=self.Vdt.get().astype(np.float32))
        if hasattr(self, "energy"):
            f.create_dataset("energy", data=self.energy.astype(np.float32))
        if hasattr(self, "number"):
            f.create_dataset("number", data=self.number.astype(np.float32))
        if hasattr(self, "times"):
            f.create_dataset("times", data=self.times.astype(np.float32))
        if hasattr(self, "spectrum"):
            f.create_dataset("spectrum",
                             data=self.spectrum.astype(np.complex64))
            f.create_dataset("omega_axis",
                             data=self.omega_axis.astype(np.float32))

        # paramsToSave = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'm', 'charT',
        #                 'charL']
        for (param, value) in self.paramContainer.getOutputParams().items():
            f.attrs[param] = value.magnitude
        for (attr, value) in kwargs.items():
            f.attrs[attr] = value
        f.attrs["t"] = self.time
        f.close()
