import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_grid(x_grid, y_grid):
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4)
    )
    im0 = axes[0].imshow(x_grid, interpolation="None")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    figure.colorbar(im0, cax=cax, orientation='vertical')

    im1 = axes[1].imshow(y_grid, interpolation="None")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    figure.colorbar(im1, cax=cax, orientation='vertical')

    plt.show()


def plot_uv_wavelengths(uv_wavelengths):

    if len(uv_wavelengths.shape) == 1:
        raise ValueError
    elif len(uv_wavelengths.shape) == 2:
        pass
    elif len(uv_wavelengths.shape) == 3:
        u_wavelengths = uv_wavelengths[:, :, 0]
        v_wavelengths = uv_wavelengths[:, :, 1]

        # NOTE: Still some problems with the colors ...
        plt.figure()
        colors = pl.cm.jet(
            np.linspace(0, 1, uv_wavelengths.shape[0])
        )
        for i in range(uv_wavelengths.shape[0]):
            plt.plot(
                u_wavelengths[i, :] * 10**-3.0,
                v_wavelengths[i, :] * 10**-3.0,
                linestyle="None",
                marker=".",
                color=colors[i]
            )
        plt.show()



def plot_visibilities(visibilities, spectral_mask=None):

    #total_number_of_channels = visibilities.shape[0]
    if len(visibilities.shape) == 1:
        raise ValueError
    elif len(visibilities.shape) == 2:
        real_visibilities = visibilities[:, 0]
        imag_visibilities = visibilities[:, 1]

        plt.figure()
        plt.plot(
            real_visibilities,
            imag_visibilities,
            linestyle="None",
            marker="."
        )

        plt.show()

    elif len(visibilities.shape) == 3:
        real_visibilities = visibilities[:, :, 0]
        imag_visibilities = visibilities[:, :, 1]

        plt.figure()
        colors = pl.cm.jet(
            np.linspace(0, 1, visibilities.shape[0])
        )
        for i in range(visibilities.shape[0]):
            plt.plot(
                real_visibilities[i, :],
                imag_visibilities[i, :],
                linestyle="None",
                marker=".",
                color=colors[i]
            )

        plt.show()


# NOTE: KEEP THIS EMPTY, BECAUSE GALPAK IS READING FROM ANOTHER GITHUB REPO LOCALLY.
# def plot_cube():
#     pass

def plot_cube(cube, ncols, cmin=None, cmax=None, xmin=None, xmax=None, ymin=None, ymax=None, cube_contours=None, figsize=None, imshow_kwargs={}, subplots_kwargs={"wspace":0.01, "hspace":0.01}):


    if cmin is None:
        cmin = 0
    if cmax is None:
        cmax = cube.shape[0]

    # ...
    cube = cube[cmin:cmax, :, :]

    # ...
    N = cube.shape[0]

    # ...
    if N % ncols == 0:
        nrows = int(N / ncols)
    else:
        nrows = int(N / ncols) + 1

    # figsize_x = 20
    # if figsize_x == 20:
    #     figsize_y = nrows * 4

    if figsize is not None:
        figsize_x, figsize_y = figsize
    else:
        figsize_x = 16
        figsize_y = 8

    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            figsize_x,
            figsize_y
        )
    )

    # vmin = 0.0
    # vmax = 0.00005
    # vmin=vmin,
    # vmax=vmax

    vmin = np.nanmin(cube)
    vmax = np.nanmax(cube)

    k = 0

    for i in range(nrows):
        for j in range(ncols):
            if k < N:
                axes[i, j].imshow(
                    cube[k, :, :],
                    cmap="jet",
                    interpolation="None",
                    vmin=vmin,
                    vmax=vmax,
                    **imshow_kwargs
                )
                if cube_contours is not None:
                    axes[i, j].contour(cube_contours[k, :, :])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                k += 1
            else:
                axes[i, j].axis("off")

    plt.subplots_adjust(
        **subplots_kwargs
    )
    plt.show()
