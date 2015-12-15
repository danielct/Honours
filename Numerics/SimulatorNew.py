"""
Module that contains classes used to model the coupled GPE using split-step
fourier method
"""
from __future__ import division
from UtilityFunctions import TanhDamping
import numpy as np
import pyfftw
import json
import os
from copy import deepcopy

WISDOM_LOCATION = os.path.join(os.path.expanduser('~'), '.wisdom', 'wisdom')
FLAG = 'FFTW_PATIENT'
# Load wisdom?
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

# TODO: Equations should be defined without the i
# TODO: Allow us to specify whether fields should be complex
# TODO: Account for discrepancy between D(k^2) and D(\nabla^2)


class Equation(object):
    """
    Class that defines a system of equations of the form
    \partial_t \psi_1 = [D_N(\ nabla^2)
    + N_i (t, \psi_1, ... , \psi_N)]\psi + C_1(t)
    ...
    \partial_t \psi_N = [D_N(\ nabla^2)
    + N_N (t, \psi_1, ... , \psi_N)]\psi + C_N(t)
    """

    def __init__(self, Ds, Ns, Cs):
        """
        Define a new system of equations. If the equation has D = 0, D should be
        input as None. The solver and stepper will then detect this and gain
        performance by not doing the relevant updates.

        Parameters:
            Ds: List of D functions. Signature for these functions is
            D( \ nabla^2)
            Ns: List of N functions. Signature for these functions is
            N(t, x, y, psi_1, ..., psi_N)
            Cs: List of C functions. Signature for these functions is C(t, x, y)
        """
        length = len(Ds)
        if length != len(Ns):
            raise ValueError("Ds is not the same length as Ns")
        if length != len(Cs):
            raise ValueError("Ds is not the same length as Cs")
        self.N_fields = length
        self.Ds = list(Ds)
        self.Ns = list(Ns)
        self.Cs = list(Cs)


# TODO: Allow us to start observations after a certain time
class Solver(object):
    """
    Solver class
    """
    def __init__(self, eqn, grid, damping='default'):
        """
        Initialise a solver.

        Parameters:
            eqn: Equation object specifying the system of equations to solve.
            grid: Grid object that specifies the grid on which to solve the
            equations

            damping: An array to use for damping. Default is a tanh function]
            that drops from 1.0 to 0 over the last 10% of each dimension. If
            None, no damping will be applied.
            None, no damping will be applied.
            """

        al = pyfftw.simd_alignment
        self.grid = deepcopy(grid)
        self.eqn = deepcopy(eqn)
        N = grid.N
        self.x, self.y = grid.getSpatialGrid()
        self.psis = [pyfftw.n_byte_align_empty((N, N), al, 'complex128') for i
                     in xrange(eqn.N_fields)]
        if damping == 'default':
            self.damping = defaultDamping(grid)
        else:
            self.damping = damping

    def solve(self, psiIFunctions, dt, t_max, method='SplitStep',
              observers=None, timeStepsPerObservation=1):
        """
        Solves the system of equations. Makes obvervations by telling the
        observers to observe. Observers store the observations.

        Parameters:
            psiIFunctions: A list of functions that specify the initial state of
            each field. If an entry is None, the corresponding field will be
            initialised to zero.
            dt: Timestep.
            t_max: The maximum time.
            method: The method to use. Must be the same as the name of a stepper
            class.
            observers: List of observer object to observe with. Defaults to None
            and no observations will be made
            timeStepsPerObservation: number of time steps per observation.
            observations will be made when
            [Number of Steps] % timeStepsPerObservation == 0

        Returns:
            A list of the final field values.
        """
        # TODO: Add support for simulating over a specific time interval
        # TODO: Let us define initial conditions with an array
        if len(psiIFunctions) != self.eqn.N_fields:
            raise ValueError("Number of initial functions is not the same as\
                             the number of fields specified by the equations")

        try:
            stepper = STEPPER_CLASSES[method](self.psis, self.grid, self.eqn,
                                              dt)
        except KeyError:
            raise ValueError("Method not recognised")

        # Initialise stepper before initialising psis. Stepper will initialise
        # FFTS, which will destroy the contents of the psis.

        for i, psiIFunction in enumerate(psiIFunctions):
            if psiIFunction is not None:
                self.psis[i][:] = psiIFunction(self.x, self.y)
            else:
                self.psis[i][:] = 0.0
        N_STEPS = int(t_max / dt)
        t = 0

        for step in xrange(N_STEPS):
            if step % timeStepsPerObservation == 0:
                if observers is not None:
                    for observer in observers:
                        observer(self.psis, t)
            stepper(t)
            if self.damping is not None:
                for psi in self.psis:
                    psi *= self.damping
            t += dt

        out = [np.copy(field) for field in self.psis]
        return out


class Stepper(object):
    def __init__(self, fields, grid, eqn, dt):
        self.KSquaredGrid = grid.getKSquaredgrid()
        self.fields = fields
        self.grid = grid
        self.x, self.y = grid.getSpatialGrid()
        self.eqn = eqn
        self.dt = dt

    def __call__(self, t):
        raise NotImplementedError("Base stepper class should not be called")


class Observer(object):
    def __init__(self,  nObservations):
        self.times = np.zeros(nObservations)

    def __call__(self, fields, t):
        raise NotImplementedError("Base observer class should not be called!")


class SplitStep(Stepper):
    # TODO: Add support for batched transforms? Won't give us any performance
    # boost unless we are modelling a two-component condensate
    def __init__(self, fields, grid, eqn, dt, N_THREADS=4):
        Stepper.__init__(self, fields, grid, eqn, dt)
        # Define fft object lists
        self.fftObjectList = [None] * eqn.N_fields
        self.ifftObjectList = [None] * eqn.N_fields

        for i, (D, psi) in enumerate(zip(eqn.Ds, self.psis)):
            if D is not None:
                self.fftObjectList[i] = pyfftw.FFTW(psi, psi, flags=[FLAG],
                                                    axes=(0, 1),
                                                    threads=N_THREADS)
                self.fftObjectList[i] = pyfftw.FFTW(psi, psi, flags=[FLAG],
                                                    axes=(0, 1),
                                                    threads=N_THREADS,
                                                    direction='FFT_BACKWARD')

        # Cache diffusive exponential factors.
        self.diffusiveExponentialFactorsHalf = [None] * eqn.N_fields
        for i, D in enumerate(eqn.Ds):
            if D is not None:
                self.diffusiveExponentialFactorsHalf[i] = np.exp(
                    D(-self.KSquaredGrid) * 0.5 * dt)

        def __call__(self, t):

            # Do the diffusive update on each field
            for (field, fft, ifft, D) in zip(self.fields, self.fftObjectList,
                                             self.ifftObjectList,
                                             self.
                                             diffusiveExponentialFactorsHalf):
                if D is not None:
                    fft()
                    field *= D
                    ifft()

            # Do the nonlinear updates.
            for (field, N, C) in zip(self.fields, self.eqn.Ns, self.eqn.Cs):
                if N is not None:
                    field *= np.exp(N(t, self.x, self.y,  *fields) * dt)
                if C is not None:
                    field += C(t, self.x, self.y) * dt

# TODO: Allow for us to provide an array representing the pump
class GPECoupled(Equation):
    """
    Class that defines an Equation instance that represents an instance of the
    normalised resovoir-coupled GPE. That is,
    \partial_t \psi = -i (\ nabla^2 + i/2 * (R * n_R - gamma_C) + g_C * |\psi|^2
        + g_R * n_R + V(r, t) ) \psi

    \partial_t n_R = P(r, t) - (gamma_R + R |\psi|^2)*n_R
    """
    __requiredParams = ['g_C', 'g_R', 'gamma_C', 'gamma_R', 'R']

    def __init__(self, params, pumpFunction, potentialFunction=None):
        """
        Initialise a new instance of the GPE.

        Parameters:
            params: A dictionary containing the parameters of the GPE. That is,
            g_C, g_R, gamma_C, gamma_R, and R

            potentialFunction: A function that defines the external potential as
            a function of spatial position and time. Has the signature
            V(t, x, y). If the potential does not depend on time, you should
            provide a function that takes t as an argument, but simply ignores
            it. If not provided, will default to zero.

            pumpFunction: A function that defines the pump as a function of
            spatial position and time. Has the signature P(t, x, y). If the
            pump does not depend on time, you should provide a function that
            takes t as an argument, but simply ignores it.
        """

        # Check we have the required parameters and assign them
        for key in self.__class__.__requiredParams:
            if key not in params:
                raise ValueError("Required Parameter %s missing" % key)
            self.__setattr__(key, params[key])

        # Use the params to create the D, N, and C functions. Pass them to
        # equation.__init__
        Ds = [None, None]
        Ns = [None, None]
        Cs = [None, None]

        Ds[0] = lambda x: 1j * x
        if potentialFunction:
            Ns[0] = lambda t, x, y, psi, n: (0.5*self.R - 1j*self.g_R) * n \
                - 0.5*self.gamma_C - 1j*self.g_C*np.absolute(psi)**2 \
                + potentialFunction(t, x, y)
        else:
            Ns[0] = lambda t, x, y, psi, n: (0.5*self.R - 1j*self.g_R) * n \
                - 0.5*self.gamma_C - 1j*self.g_C*np.absolute(psi)**2
        Ns[1] = lambda t, x, y, psi, n: -(self.gamma_R +
                                          self.R * np.absolute(psi)**2)*n
        Cs[1] = lambda t, x, y, psi, n: pumpFunction(t, x, y)

        Equation.__init__(self, Ds, Ns, Cs)


class EnergyObserver(Observer):
    """
    Energy observer.
    """
    #
    def __init__(self, nObservations, eqn, grid):
        """

        """
        Observer.__init__(self, nObservations)
        self.energies = np.zeros_like(self.times)
        self.g_C = eqn.g_C
        self.g_R = eqn.g_R
        self.dx = grid.dx_unscaled

    def __call__(self, fields, t):
        """
        Only works if equation is GPECoupled. Doesn't take potential into
        account.
        """

        psi = fields[0]
        n = fields[1]
        density = np.absolute(psi) ** 2
        gradx = np.gradient(psi)[0]
        normFactor = density.sum()
        self.times[self.count] = t
        self.energies[self.count] = (0.25 * np.gradient(gradx)[0]
                                     - 0.5 * gradx * gradx.conj()
                                     - (self.g_C * density
                                        + self.g_R * n) * density).sum() \
        / normFactor

# Begin damping helper functions. Each damping function takes a grid as the
# argument and returns an array that represents the damping mask
def defaultDamping(grid):
    max_XY = grid.max_XY_scaled
    x, y = grid.getSpatialGrid()
    return TanhDamping(max_XY).unscaledFunction()(x, y)

STEPPER_CLASSES = {'SplitStep': SplitStep}
DAMPING_FUNCTIONS = {'default': defaultDamping}
