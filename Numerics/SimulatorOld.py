"""
Module that contains classes used to model the coupled GPE using split-step
fourier method
"""
from __future__ import division
import numpy as np
from numpy import fft
import pyfftw
import UtilityFunctions
import json
import os

# TODO: Fix constants so that we use the pint definition
# TODO: Or use a better units package

WISDOM_LOCATION = os.path.join(os.path.expanduser('~'), '.wisdom', 'wisdom')


class ParameterContainer(object):
    """
    This class serves as a container for the values of the parameters used in an
    experiment, and provides access to the scaling factors.
    """
    __requiredArgs = ['R', 'g_C', 'gamma_C', 'm']
    __allowedArgs = ['R', 'g_C', 'g_R', 'gamma_R', 'gamma_C', 'm',
                     's', 'r', 'charT', 'charL', 'k']
    __outArgs = ['R', 'g_C', 'g_R', 'gamma_C', 'gamma_R', 'Pth']
    import pint
    ureg = pint.UnitRegistry()
    __m_e = ureg.electron_mass.to_base_units().magnitude
    __hbar = ureg.hbar.to_base_units().magnitude
    # TODO: DOcumentation

    def __init__(self, **kwargs):
        """
        Initialises all the parameters and computes the scaling factors. If gR
        is not provided, it will default to r * gc.
        m is in units of electron mass. If not provided, it will default to
        10e-4 * m_e.
        If gamma_R is not provided, it will default to s * gamma_C
        All parameters should be given in SI units

        :Parameters:
            **kwargs:
                The following are all required:
                    R : Stimulated scattering rate
                    g_C : Polariton-polariton interaction strength
                    gamma_C : Polariton relaxation rate
                    m : Polariton effective mass, in units of the electron mass
                The following are (conditionally) optional:
                    g_R : Polariton-exciton interaction strength. Default value
                    is r * g_C. Exactly one of g_R and r must be provided.
                    gamma_R : Exciton relaxtion rate. Default value is
                    s * gamma_C. Exactly one of s and gamma_C must be provided.
                    r : the ratio g_R / g_C. Exactly one of g_R or r must be
                    provided.
                    s : the ratio gamma_R / gamma_C. Exactly one of gamma_R or
                    s must be provided.
                    k:
                    charT: a characteristic time to scale by. If not provded,
                    defaults to (gamma_C) ^ -1
                    charL: a charactersitic length to scale by. If not provided,
                    defaults to ( hbar * charT / (2 * m * gamma_C) )^1/2.
        """
        self.__checkArgs(kwargs)

        # Read in the keyword arguments
        for (k, v) in kwargs.items():
            assert k in self.__class__.__allowedArgs, "Invalid Argument %s" % k
            setattr(self, k, v)

        # Set mass. We read in mass in units of electron mass for convenience,
        # but it must be converted to SI units.
        self.m = self.__class__.__m_e * self.m

        # Read in k or set to default
        self.k = kwargs.get('k', self.__class__.__hbar**2 / (2 * self.m))

        # Deal with unspecified g_R and gamma_R
        if 'g_R' not in kwargs:
            self.g_R = self.r * self.g_C
        if 'gamma_R' not in kwargs:
            self.gamma_R = self.s * self.gamma_C
        # Define our characteristic length, time, and energy scales.
        # If t' is the (nondimensional) time variable used in the
        # nondimensionalised GPE, then t = charT * t'. For example, R' is the
        # stimulated scattering rate used in the normalised GPE, so R = charR *
        # R'. If they are not provided, we will set them to the default, which
        # is the scaling that makes k=1 and gamma'_C = 1.

        self.charT = kwargs.get('charT', 1.0 / self.gamma_C)
        self.charL = kwargs.get('charL', np.sqrt((self.__class__.__hbar
                                                  * self.charT)
                                                 / (2.0 * self.m)))
        # TODO: check these
        # A characteristic energy
        self.charU = self.__class__.__hbar / self.charT
        self.charg = (self.__class__.__hbar * self.charL**2) / self.charT
        self.charR = self.charL ** 2 / self.charT
        self.charGamma = 1.0 / self.charT
        self.chark = (self.__class__.__hbar * self.charL**2) / self.charT
        # This may not be required - the P term in the GPE is phenomonological,
        # and the experimentalist probably only knows it in terms of Pth
        self.charP = 1.0 / (self.charT * self.charL ** 2)
        self.charE = self.__class__.__hbar / self.charT

        # Scaled parameters - these are the ones to used in the
        # nondimensionalised GPE
        self.g_C_scaled = self.g_C / self.charg
        self.g_R_scaled = self.g_R / self.charg
        self.gamma_C_scaled = self.gamma_C / self.charGamma
        self.gamma_R_scaled = self.gamma_R / self.charGamma
        self.k_scaled = self.k / self.chark
        self.R_scaled = self.R / self.charR
        # Compute threshold pump power for the normalised GPE.
        self.Pth_scaled = ((self.gamma_R_scaled * self.gamma_C_scaled)
                           / self.R_scaled)
        # Compute threshold pump power for unnormalised GPE. We can get this
        # from the scaled one.
        self.Pth = self.charP * self.Pth_scaled

    # TODO: FIX

    # def update(self, **kwargs):
    #     """
    #     Updates the parameters given in keyword arguments and recomputes the
    #     scaling factors.

    #     Care must be taken when using this function, as it does not respect
    #     the way in which parameters were originally defined. For example, if
    #     g_C and r were initially provided, updating g_C will *not* cause g_R
    #     to be updated. If g_C and g_R were originally provided, and r is
    #      updated, g_R will be updated to r * g_C

    #     :Parameters:
    #         **kwargs:
    #             May contain any of the keyword arguments allowed in __init__,
    #             subject to the consistency requirements concerning g_R and r,
    #             and gamma_R and s.

    #     """
    #     # Get the new values of the required arguments. We do this by copying
    #     # the current values of the required arguments into a dictionary,
    #     # an then updating from/adding the keyword arguments, and then using
    #     # these to initialise again.

    #     # Add required arguments and their current values to our new keyword
    #     # arguments.
    #     newkwArgs = {k: self.__dict__.get(k)
    #                  for k in self.__class__.__requiredArgs}
    #     # Update required arguments and add keyword arguments that were just
    #     # provided
    #     for (k, v) in kwargs.items():
    #         newkwArgs[k] = v
    #     # There's no need to check that all the required arguments are present
    #     # as we already had them.
    #     # Check consistency here, so that the trace will be cleaner if an
    #     # exception is raised.
    #     self.__checkArgs(newkwArgs)
    #     self.__init__(newkwArgs)

    def __checkArgs(self, kwargs):
        """
        Checks the consistency and validity of the keyword arguments. Raises
        errors if the system is not completely specified, or if the arguments
        provided are not consistent (ie, if gamma_c, gamma_R and r are all
        provided).
        """
        for arg in self.__class__.__requiredArgs:
            if arg not in kwargs:
                raise ValueError("Essential keyword argument %s missing" % arg)

        # User has to provide s xor gamma_R.
        if not (('s' in kwargs) ^ ('gamma_R' in kwargs)):
                raise ValueError("Exactly one of s and gamma_R must be "
                                 "provided.")
        # User has to provide s xor gamma_R
        if not (('r' in kwargs) ^ ('g_R' in kwargs)):
                raise ValueError("Exactly one of r and g_R must be "
                                 "provided.")

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
        outKeys = ['R', 'gamma_C', 'gamma_R', 'g_C', 'g_R', 'Pth', 'k']
        out = {key: self.__dict__[key + "_scaled"] for key in outKeys}
        return out


class Grid(object):
    """
    A class that contains derived GPE parameters and uses them to create scaled
    and unscaled grids.
    """
    # TODO: Allow us to provide a characteristic size in scaled units.
    def __init__(self, charLength, N=1024, **kwargs):
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

            **kwargs: Arguments that define the size of the grid. May contain:
                max_XY: the maximum spatial extent of the grid, in SI units. If
                provided, the grid will run from -max_XY to max_XY in both
                dimensions.

                charSize: A characteristic size, in SI units. If provided, the
                grid will run from - scaling * charSize to scaling * charSize
                in both
                dimensions. Using charSize to contruct a grid allows us to
                automatially construct grids that are large enough to to contain
                the relevant regions of our pumps and traps. For example, one
                may define a gaussian pump, and then use its FWHM radius to
                automatically get a grid which completely contains the pump
                spot.

                scaling: Optional scaling argument if a charSize is provided.
                default value  is 5.0, so the grid will be 5 times as big as the
                relevant object.
        """
        # Check that the kwargs are consistent. max_XY xor charSize must be
        # provided.
        if not (('max_XY' in kwargs) ^ ('charSize' in kwargs)):
            raise ValueError("Exactly one of max_XY or charSize "
                             "must be provided")
        # Check that N is even. If N is not even there may be a problem with
        # setting up the K grid.
        if N % 2 != 0:
            raise ValueError("N must be even")
        # Check that N is a power of 2
        if bin(N).rfind('1') != 2:
            raise Warning("N is not a power of 2. FFTs may be slow!")
        if 'charSize' in kwargs:
            scaling = kwargs.get('scaling', 5.0)
            self.max_XY_unscaled = scaling * self.charSize
        else:
            self.max_XY_unscaled = kwargs['max_XY']
        self.N = N
        self.charL = charLength
        self.max_XY_scaled = self.max_XY_unscaled / self.charL
        # Unscaled grid. x and y values are in SI units.
        self.x_axis_unscaled = np.linspace(- self.max_XY_unscaled,
                                           self.max_XY_unscaled, num=N)
        self.dx_unscaled = np.absolute(self.x_axis_unscaled[0]
                                       - self.x_axis_unscaled[1])
        self.dx_scaled = self.dx_unscaled / self.charL
        self.k_axis_unscaled = (((2.0 * np.pi) / (2.0 * self.max_XY_unscaled)) *
                                fft.fftshift(np.linspace(-N/2, N/2 - 1, num=N)))
        # Grid in units of the characteristic length provided in params
        self.x_axis_scaled = np.linspace(
            -self.max_XY_scaled, self.max_XY_scaled, num=N)
        self.k_axis_scaled = (((2.0 * np.pi) / (2.0 * self.max_XY_scaled)) *
                              fft.fftshift(np.linspace(-N/2, N/2 - 1, num=N)))

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
            return np.meshgrid(self.k_axis_scaled, self.k_axis_scaled)
        else:
            return np.meshgrid(self.k_axis_unscaled, self.k_axis_unscaled)

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

# TODO: Make FFTW objects private access
# TODO: Use the new version of GPE with k in the RK4 methods (if you actually
# want to use RK4)


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
    __requiredParams = ['g_C', 'g_R', 'gamma_C', 'gamma_R', 'R', 'Pth', 'k']

    def __init__(self, spatialGrid, kGrid, damping='default',
                 FFTW_METHOD='FFTW_PATIENT', N_THREADS=6):
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

        # Load any existing wisdom
        try:
            wisdomFile = open(WISDOM_LOCATION, 'r+')
            importStatus = pyfftw.import_wisdom(json.load(wisdomFile))
            print "Wisdom found"
            if not np.array(importStatus).all():
                print "Wisdom not loaded correctly"
                # raise Warning("Wisdom not loaded correctly.")
            wisdomFile.close()
        except IOError:
            print "Wisdom not present"
            # raise Warning("Wisdom not present.")

        self.x, self.y = spatialGrid
        self.kx, self.ky = kGrid
        self.K = self.kx ** 2 + self.ky ** 2
        # This is already scaled because we obtained it from the scaled grid.
        self.max_XY = np.abs(self.x[-1, -1])
        self.N = self.x.shape[0]
        self.N_THREADS = N_THREADS
        self.time = 0
        # TODO: Allow for rectangular grids.
        assert self.x.shape == self.y.shape, "Spatial grids are not the same\
               shape"
        assert self.kx.shape == self.ky.shape, "k grids are not the same shape"
        assert self.x.shape == self.kx.shape, "Spatial grids are not the same\
               shape as k grid"
        assert self.x.shape == (self.N, self.N), 'Grid must be square.'

        if damping == 'default':
            tanhDamping = UtilityFunctions.RadialTanhDamping(self.max_XY)
            # We can use the unscaled function here because max_XY is already
            # scaled.
            # A damping mask
            self.damping = tanhDamping.unscaledFunction()(self.x, self.y)

        # Set up fftw objects
        # Optimal alignment for this CPU
        self.al = pyfftw.simd_alignment
        self.psi = pyfftw.n_byte_align_empty((self.N, self.N), self.al,
                                             'complex128')
        flag = FFTW_METHOD
        self.fft_object = pyfftw.FFTW(self.psi, self.psi, flags=[flag],
                                      axes=(0, 1), threads=N_THREADS)
        self.ifft_object = pyfftw.FFTW(self.psi, self.psi, flags=[flag],
                                       axes=(0, 1), threads=N_THREADS,
                                       direction='FFTW_BACKWARD')

        # Save pyfftw's newfound wisdom
        f = open(WISDOM_LOCATION, 'w+')
        json.dump(pyfftw.export_wisdom(), f)
        f.close()

    # TODO: Make the different solutioj methods different methods.
    def solve(self, params, pumpFunction, psi0Function=None,
              potentialFunction=None, dt=0.1, T_MAX=50, continuing=False,
              method='split-step', recordEnergy=True, stepsPerObservation=10):
        """
        Solve the GPE for the given conditions. Records the final condensate
        wavefunction and density in the GPESolver object.

        Parameters:

            params: A dictionary containing at leas the parameters of the
            normalised GPE equation. That is, g_C, g_R, gamma_C, gamma_R, R, and
            Pth

            dt: the time increment per timestep. In units of the characteristic
            time defined in ParameterContainer

            T_MAX: the maximum time. In units of the characteristic time defined
            in ParameterContainer.

            pumpFunction: A function that describes the pump power in units of
            the threshold power.

            potentialFunction: A function that describes the potential in units
            of the characteristic energy scale. Default is 0

            psi0Function: A function that describes the initial wavefunction in
            units of inverse characteristic length. Default is a random seed
            whose real and imaginary parts are both drawn from a normal
            distributions with a small (10e-2) mean and (10e-3) standard
            deviation.

            continue: Flag used to continue a simulation. If true, the inital
            self.psiFinal will be taken as the initial wavefunction, and the
            simulation will be continued in steps of dt until T_MAX is reached
        """

        # TODO  Add support for making videos, recording the energy, etc. as
        # the need arises

        # Check we have the required parameters and assign them
        for key in self.__class__.__requiredParams:
            if key not in params:
                raise ValueError("Required Parameter %s missing" % key)
            self.__setattr__(key, params[key])

        # Check that we are not continuing and providing psi0
        if continuing and psi0Function:
            raise ValueError("Can't continue a simulation and provide psi0")

        # Assign the potential, pump, etc.
        P = pumpFunction(self.x, self.y)

        if not continuing:
            self.time = 0.0
            if not psi0Function:
                psi0 = (np.abs(np.random.normal(size=(self.N, self.N),
                                                scale=10e-5, loc=10e-4))
                        + 0.0j*np.random.normal(size=(self.N, self.N),
                                                scale=10e-4, loc=10e-3))
            else:
                psi0 = psi0Function(self.x, self.y)
        else:
            psi0 = np.copy(self.psiFinal)

        if not potentialFunction:
            potential = 0.0
        else:
            potential = potentialFunction(self.x, self.y)

        # Carefully copy psi0 into psi. We want to be certain that we keep the
        # alignment
        self.psi[:] = 1
        self.psi *= psi0
        # Check psi is aligned
        assert pyfftw.is_n_byte_aligned(self.psi, self.al)

        n = np.zeros_like(self.psi, dtype=np.float64) + 1.0

        # Get kinetic factor, initial density
        # TODO: Is the factor of 1/2 correct?
        kineticFactorHalf = np.exp(-1.0j * self.k * self.K * dt / 2.0)
        currentDensity = np.absolute(psi0) ** 2

        # Do the simulation
        N_TIMESTEPS = int((T_MAX - self.time) // dt)

        # Set up array to record energy
        if recordEnergy:
            self.energy = np.zeros(N_TIMESTEPS // stepsPerObservation + 1)
            self.energyTimes = np.zeros_like(self.energy)

        if method == 'split-step':
            for step in xrange(N_TIMESTEPS):
                # Implementing split-step method

                # Record energy
                if step % stepsPerObservation == 0 and recordEnergy:
                    density = np.absolute(self.psi) ** 2
                    gradx = np.gradient(self.psi)[0]
                    normFactor = density.sum()
                    self.energyTimes[step // stepsPerObservation] = self.time
                    self.energy[step // stepsPerObservation] = \
                        -(0.25 * np.gradient(
                            np.gradient(density)[0])[0]
                            - 0.5 * np.absolute(gradx) ** 2
                            - (self.g_C * density + self.g_R * n)
                            * density).sum() / normFactor

                # Update wavefunction and resovoir, record density

                # Take fft, multiply by kinetic factor and then take inverse fft
                self.fft_object()
                self.psi *= kineticFactorHalf
                self.ifft_object()

                currentDensity = np.absolute(self.psi) ** 2

                expFactorExciton = np.exp(- (self.gamma_R
                                             + self.R * currentDensity) * dt)
                n *= expFactorExciton
                n += P * dt

                expFactorPolariton = np.exp((n*(0.5 * self.R - 1.0j * self.g_R)
                                             - 0.5 * self.gamma_C
                                             - 1.0j * self.g_C * currentDensity
                                             - 1.0j * potential) * dt)

                # Do the nonlinear update, take FFT, do kinetic update, and then
                # take the inverse fft
                self.psi *= expFactorPolariton
                self.fft_object()
                self.psi *= kineticFactorHalf
                self.ifft_object()
                self.psi *= self.damping
                n *= self.damping
                self.time += dt
                if step % 100 == 0:
                    print("Time = %f" % (self.time))

        elif method == 'RK4':
            def nonLinearUpdateN(n, P, density, dt):
                """
                Updates n (in place) to M(n, P, density, dt)
                """
                n *= - (self.gamma_R + self.R * density) * dt
                n += P * dt

            def nonLinearUpdatePsi(psi, potential, density, n, dt):
                psi *= -1j * dt * (n * (0.5j * self.R + self.g_R) -
                                   0.5j * self.gamma_C
                                   + self.g_C * density + potential)

            def diffusionUpdate(psi, psifft, psiifft, expFactor):
                psifft()
                psi *= expFactor
                psiifft()

            # Set up psiK nK, n0
            kineticFactorRK4 = np.exp(-1.0j * self.K * dt / 2.0)
            psiK = pyfftw.n_byte_align_empty((self.N, self.N), self.al,
                                             'complex128')
            n0 = np.zeros_like(n)
            nK = np.zeros_like(n)
            nkStored = np.zeros_like(n)
            psiI = np.zeros_like(self.psi)
            dK = np.absolute(self.psi) ** 2
            # Set up an fft object for psi_K
            fft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                         flags=self.fft_object.flags,
                                         axes=(0, 1), threads=self.N_THREADS)
            ifft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                          flags=self.fft_object.flags,
                                          axes=(0, 1), threads=self.N_THREADS,
                                          direction="FFTW_BACKWARD")
            psiK[:] = self.psi
            assert pyfftw.is_n_byte_aligned(self.psi, self.al)
            assert pyfftw.is_n_byte_aligned(psiK, self.al)
            for step in xrange(N_TIMESTEPS):
                n0[:] = n
                nK[:] = n
                psiK[:] = self.psi
                dK[:] = np.absolute(self.psi) ** 2
                assert pyfftw.is_n_byte_aligned(psiK, self.al)

                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                psiI[:] = self.psi

                # K=1 step
                # Updates on psiK and nK

                nonLinearUpdatePsi(psiK, potential, dK,
                                   nK, dt)
                nonLinearUpdateN(nK, P, dK, dt)
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                # Update the next value of psi and n
                self.psi += psiK / 6
                n += nK / 6
                # Update psik and nK. Store things for the next step
                psiK /= 2
                psiK += psiI
                nK /= 2
                nK += n0

                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=2 step
                # Do updates
                nonLinearUpdatePsi(psiK, potential, dK, nK, dt)
                nonLinearUpdateN(nK, P, dK, dt)
                # Update the next value of psi and n
                self.psi += psiK / 3
                n += nK / 3
                # Update psiK and nK. Store things for the next step
                psiK /= 2
                psiK += psiI
                nK /= 2
                nK += n0

                dK[:] = np.absolute(psiK) ** 2

                # K=3 step
                # Do the updates
                nonLinearUpdatePsi(psiK, potential, dK,
                                   nkStored, dt)
                nonLinearUpdateN(nK, P, dK, dt)
                # Update the next value of psi and n
                self.psi += psiK / 3
                n += nK / 3
                # Update psiK and nK. Store things for the next step
                psiK += psiI
                nK += n0
                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=4 step
                # Do the updates
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                nonLinearUpdatePsi(psiK, potential, dK, nK, dt)

                nonLinearUpdateN(nK, P, dK, dt)
                # Transform psi back
                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                # Update the next value of psi and n
                self.psi += psiK / 6
                n += nK / 6

                # Apply damping
                self.psi *= self.damping
                n *= self.damping
                print("Time: %f" % self.time)

        elif method == "RK4Exact":
            def updateNtoF(n, density, dt):
                """
                Updates n to F.
                """
                # We have to do the assignment like this in order to actually
                # change n
                n[:] = np.exp(- (self.gamma_R + self.R * density) * dt)

            def nonLinearUpdateF(F, n0, P, density, dt):
                """
                Updates F (in place) to F **(0.5) * n0 + P*dt/2
                """
                F **= 0.5
                F *= n0
                F += 0.5 * P * dt

            def nonLinearUpdatePsi(psi, potential, density, n, dt):
                psi *= -1j * dt * (n * (0.5j * self.R + self.g_R) -
                                   0.5j * self.gamma_C
                                   + self.g_C * density + potential)

            def diffusionUpdate(psi, psifft, psiifft, expFactor):
                psifft()
                psi *= expFactor
                psiifft()

            # Set up psiK nK, n0
            kineticFactorRK4 = np.exp(-1.0j * self.K * dt / 2.0)
            psiK = pyfftw.n_byte_align_empty((self.N, self.N), self.al,
                                             'complex128')
            n0 = np.zeros_like(n)
            nK = np.zeros_like(n)
            psiI = np.zeros_like(self.psi)
            dK = np.absolute(self.psi) ** 2
            # Set up an fft object for psi_K
            fft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                         flags=self.fft_object.flags,
                                         axes=(0, 1), threads=self.N_THREADS)
            ifft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                          flags=self.fft_object.flags,
                                          axes=(0, 1), threads=self.N_THREADS,
                                          direction="FFTW_BACKWARD")
            psiK[:] = self.psi
            assert pyfftw.is_n_byte_aligned(self.psi, self.al)
            assert pyfftw.is_n_byte_aligned(psiK, self.al)
            for step in xrange(N_TIMESTEPS):
                n0[:] = n
                nK[:] = n
                #
                n[:] = 0
                psiK[:] = self.psi
                dK[:] = np.absolute(self.psi) ** 2
                assert pyfftw.is_n_byte_aligned(psiK, self.al)

                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                psiI[:] = self.psi

                # K=1 step
                # Updates on psiK and nK

                nonLinearUpdatePsi(psiK, potential, dK,
                                   nK, dt)
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                updateNtoF(nK, dK, dt)
                # Update the next value of psi and n
                self.psi += psiK / 6
                n += (nK*n0 + P*dt) / 6
                # Update psik and nK. Store things for the next step
                psiK /= 2
                psiK += psiI
                nonLinearUpdateF(nK, n0, P, dK, dt)

                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=2 step
                # Do updates
                nonLinearUpdatePsi(psiK, potential, dK, nK, dt)
                updateNtoF(nK, dK, dt)
                # Update the next value of psi and n
                self.psi += psiK / 3
                n += (nK*n0 + P*dt) / 3
                # Update psiK and nK. Store things for the next step
                psiK /= 2
                psiK += psiI
                nonLinearUpdateF(nK, n0, P, dK, dt)

                dK[:] = np.absolute(psiK) ** 2

                # K=3 step
                # Do the updates
                nonLinearUpdatePsi(psiK, potential, dK,
                                   nK, dt)
                updateNtoF(nK, dK, dt)
                # For the next step, the derivatives will be evaluated at the
                # end of the timestep. So we need to update nK accordingly. This
                # value of nK is also the one to use to update n
                nK *= n0
                nK += P*dt
                # Update the next value of psi and n
                self.psi += psiK / 3
                n += nK / 3
                # Update psiK and nK. Store things for the next step
                # No need to update nK. Explained above
                psiK += psiI

                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=4 step
                # Do the updates
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                nonLinearUpdatePsi(psiK, potential, dK, nK, dt)

                updateNtoF(nK, dK, dt)
                # Can update nK to the value at the end of the step as we don't
                # need to store the half-step estimate since we don't have to do
                # any more steps after this.
                nK *= n0
                nK += P*dt
                # Transform psi back
                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                # Update the next value of psi and n
                self.psi += psiK / 6
                n += nK / 6

                # Apply damping
                self.psi *= self.damping
                n *= self.damping
                print("Time: %f" % self.time)

        elif method == 'RK4LessExact':
            def nonLinearUpdateN(n, P, density, dt):
                """
                Updates n (in place) to M(n, P, density, dt)
                """
                n *= np.exp(- (self.gamma_R + self.R * density) * dt)
                n += P * dt

            def nonLinearUpdatePsi(psi, potential, density, n, dt):
                psi *= -1j * dt * (n * (0.5j * self.R + self.g_R) -
                                   0.5j * self.gamma_C
                                   + self.g_C * density + potential)

            def diffusionUpdate(psi, psifft, psiifft, expFactor):
                psifft()
                psi *= expFactor
                psiifft()

            # Set up psiK nK, n0
            kineticFactorRK4 = np.exp(-1.0j * self.K * dt / 2.0)
            psiK = pyfftw.n_byte_align_empty((self.N, self.N), self.al,
                                             'complex128')
            dK = np.zeros_like(n)
            psiI = np.zeros_like(self.psi)
            # Set up an fft object for psi_K
            fft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                         flags=self.fft_object.flags,
                                         axes=(0, 1), threads=self.N_THREADS)
            ifft_objectpsiK = pyfftw.FFTW(psiK, psiK,
                                          flags=self.fft_object.flags,
                                          axes=(0, 1), threads=self.N_THREADS,
                                          direction="FFTW_BACKWARD")
            psiK[:] = self.psi
            assert pyfftw.is_n_byte_aligned(self.psi, self.al)
            assert pyfftw.is_n_byte_aligned(psiK, self.al)
            for step in xrange(N_TIMESTEPS):
                # Record energy
                if step % stepsPerObservation == 0 and recordEnergy:
                    density = np.absolute(self.psi) ** 2
                    gradx = np.gradient(self.psi)[0]
                    normFactor = density.sum()
                    self.energyTimes[step // stepsPerObservation] = self.time
                    self.energy[step // stepsPerObservation] = \
                        -(0.25 * np.gradient(
                            np.gradient(density)[0])[0]
                            - 0.5 * np.absolute(gradx) ** 2
                            - (self.g_C * density + self.g_R * n)
                            * density).sum() / normFactor

                psiK[:] = self.psi
                dK[:] = np.absolute(psiK) ** 2

                assert pyfftw.is_n_byte_aligned(psiK, self.al)

                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                psiI[:] = self.psi

                # K=1 step
                # Updates on psiK and nK

                nonLinearUpdatePsi(psiK, potential, dK,
                                   n, dt)
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                # Update the next value of psi and n
                self.psi += psiK / 6
                # Update psik and nK. Store things for the next step
                psiK /= 2
                psiK += psiI

                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=2 step
                # Do updates
                nonLinearUpdatePsi(psiK, potential, dK, n, dt)
                # Update the next value of psi and n
                self.psi += psiK / 3
                # Update psiK and nK. Store things for the next step
                psiK /= 2
                psiK += psiI

                dK[:] = np.absolute(psiK) ** 2

                # K=3 step
                # Do the updates
                nonLinearUpdatePsi(psiK, potential, dK,
                                   n, dt)
                # Update the next value of psi and n
                self.psi += psiK / 3
                # Update psiK and nK. Store things for the next step
                psiK += psiI
                dK[:] = np.absolute(psiK) ** 2
                self.time += dt / 2

                # K=4 step
                # Do the updates
                diffusionUpdate(psiK, fft_objectpsiK, ifft_objectpsiK,
                                kineticFactorRK4)
                nonLinearUpdatePsi(psiK, potential, dK, n, dt)

                # Transform psi back
                diffusionUpdate(self.psi, self.fft_object, self.ifft_object,
                                kineticFactorRK4)
                # Update the next value of psi and n
                self.psi += psiK / 6

                dK[:] = np.absolute(self.psi) ** 2
                # Do "exact" update on n
                nonLinearUpdateN(n, P, dK, dt)

                # Apply damping
                self.psi *= self.damping
                n *= self.damping
                print("Time: %f" % self.time)

        else:
            raise ValueError("Invalid method")

        self.nFinal = np.copy(n)
        self.psiFinal = np.copy(self.psi)
