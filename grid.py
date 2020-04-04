import numpy as np
import matplotlib.pyplot as plt
from astropy import units

from plot_utils import *

def grid_2d_in_radians(n_pixels, pixel_scale):

    x = np.linspace(
        -n_pixels * pixel_scale / 2.0 + pixel_scale / 2.0,
        +n_pixels * pixel_scale / 2.0 - pixel_scale / 2.0,
        n_pixels)
    y = np.linspace(
        +n_pixels * pixel_scale / 2.0 - pixel_scale / 2.0,
        -n_pixels * pixel_scale / 2.0 + pixel_scale / 2.0,
        n_pixels)
    x_grid, y_grid = np.meshgrid(
        x, y
    )

    x_grid_in_radians = x_grid * (units.arcsec).to(units.rad)
    y_grid_in_radians = y_grid * (units.arcsec).to(units.rad)

    # plt.figure()
    # plt.imshow(x_grid_in_radians, interpolation="None")
    # #plt.imshow(y_rad, interpolation="None")
    # plt.colorbar()
    # plt.show()

    return x_grid_in_radians, y_grid_in_radians


# class grid:
#     def __init__(self, n_pixels, pixel_scale):
#
#         x = np.linspace(
#             -n_pixels * pixel_scale / 2.0 + pixel_scale / 2.0,
#             +n_pixels * pixel_scale / 2.0 - pixel_scale / 2.0,
#             n_pixels)
#         y = np.linspace(
#             +n_pixels * pixel_scale / 2.0 - pixel_scale / 2.0,
#             -n_pixels * pixel_scale / 2.0 + pixel_scale / 2.0,
#             n_pixels)
#         self.x_grid, self.y_grid = np.meshgrid(
#             x, y
#         )
#
#         self._grid = np.dstack((self.y_grid, self.x_grid))
#
#
#
#     # def uniform(self, n_pixels, pixel_scale):
#     #     pass
#
#     # @property
#     # @array_util.Memoizer()
#     # def in_radians(self):
#     #     return (self * np.pi) / 648000.0


if __name__ == "__main__":
    pass
    #grid_2d_in_radians(n_pixels=20, pixel_scale=0.1)
