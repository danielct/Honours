import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot3D(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, linewidth=0, cmap=cm.jet)
    # fig.colorbar(surf, shrink=0.5)
    plt.show()
