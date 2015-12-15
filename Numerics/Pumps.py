import numpy as np


class SpatialFunction(object):
    """
    Not to be used.
    Parent class for spatial functions such as the pump and potential.
    Spatial functions are required to provide a function that corresponds to
    the spatial function. Eg, for a pump, the function would take an x grid and
    y grid and output the pump power density function.
    Should also provide a characteristic size so that the grid extent may be
    scaled to it.
    """
    # TODO: Write the above more clearly

    def __init__(self):
        raise NotImplementedError()


class gaussianPump(SpatialFunction):
    """
    A Gaussian pump
    """
    # TODO: Allow for no scaling.
    # TODO: Allow for ellispoidal spot shape
    def __init__(self, sigma, P0, Pth):
        """
        Make a Gaussian pump with maximum power P0 * Pth, where Pth is the
        threshold value of P in the normalised GPE.

        Parameters:
            sigma: value of sigma for the Gaussian. Should be in SI units
            P0: Maximum pump power. In units of Pth.
            Pth: The value of the threshold power in the scaled GPE.
        """
        self.charSize = 2.0 * np.sqrt(2 * np.ln(2)) * sigma
        self.function = lambda x, y: (P0 * Pth *
                                      np.exp(-0.5 * (x**2 + y**2) / sigma**2))
