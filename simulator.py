import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from visibilities import load_uv_wavelengths_from_fits
from grid import grid_2d_in_radians
from transformer import Transformer
from model import model
from plot_utils import plot_cube, plot_visibilities

sys.path.insert(
    0,
    os.environ["Galpak3D"]
)
import galpak


class simulator:
    def __init__(self, transformer, model):

        self.transformer = transformer
        self.model = model


if __name__ == "__main__":

    filename = "./uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    uv_wavelengths = np.dstack((u_wavelengths, v_wavelengths))
    #print(u_wavelengths.shape);exit()



    n_pixels = 50
    total_pixels = int(n_pixels * n_pixels)
    pixel_scale = 0.1
    x_grid_in_radians, y_grid_in_radians = grid_2d_in_radians(
        n_pixels=n_pixels, pixel_scale=pixel_scale
    )
    grid_1d_in_radians = np.array([
        np.ndarray.flatten(y_grid_in_radians),
        np.ndarray.flatten(x_grid_in_radians)
    ]).T
    # plot_grid(x_grid=x_grid_in_radians, y_grid=y_grid_in_radians)
    # exit()

    transformer = Transformer(
        uv_wavelengths=uv_wavelengths,
        grid=grid_1d_in_radians,
        preload_transform=True
    )
    #print(transformer.cube_shape);exit()

    theta = [
        int(n_pixels / 2.0),
        int(n_pixels / 2.0),
        transformer.n_channels / 2.0,
        0.25,
        0.75 / pixel_scale,
        50.0,
        65.0,
        0.2 / pixel_scale,
        300.0,
        50.0
    ]
    galaxy_parameters = galpak.GalaxyParameters.from_ndarray(theta)
    print(galaxy_parameters);exit()

    dv = 50.0
    model = Model(shape=transformer.cube_shape)
    cube = model.create_cube(
        galaxy_parameters=galaxy_parameters, dv=dv
    )
    cube=cube.data
    # plot_cube(cube.data, ncols=8)

    model_cube_reshaped = cube.reshape(
        cube.shape[0], total_pixels
    )

    visibilities = transformer.visibilities_from_cube(
        cube_in_1d=model_cube_reshaped
    )
    #print(visibilities.shape)
    #plot_visibilities(visibilities=visibilities)

    fits.writeto(
        "./visibilities.fits",
        data=visibilities,
        overwrite=True
    )
