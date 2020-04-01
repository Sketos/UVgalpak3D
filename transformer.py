import numpy as np


# from autoarray.util import transformer_util
# #from autoarray.structures import arrays, visibilities as vis
# from astropy import units
# #from scipy import interpolate
# #from pynufft import NUFFT_cpu
#

#
# #from autoarray import exc
#

from visibilities import load_uv_wavelengths_from_fits
from grid import grid_2d_in_radians

from transformer_utils import *
from plot_utils import *


class Transformer:
    def __init__(self, uv_wavelengths, grid=None, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths
        #print(self.uv_wavelengths.shape); exit()

        self.n_channels = self.uv_wavelengths.shape[0]

        # NOTE: Not nessecaary
        # self.u_wavelengths = self.uv_wavelengths[:, :, 0]
        # self.v_wavelengths = self.uv_wavelengths[:, :, 1]
        #
        # self.u_wavelengths_0 = self.u_wavelengths[0]
        # self.v_wavelengths_0 = self.v_wavelengths[0]

        self.uv_wavelengths_0 = self.uv_wavelengths[0]
        #print(self.uv_wavelengths_0.shape)

        self.uv_wavelengths_step = self.uv_wavelengths[0, :, :] - self.uv_wavelengths[1, :, :]


        # NOTE: At this point the grid is in radians. This will be made into a function that will hold both grid and grid.in_radians
        self.grid = grid
        #print(self.grid.shape);exit()


        self.total_visibilities = uv_wavelengths.shape[0]

        # self.total_image_pixels = grid.shape_1d

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = preload_real_transforms(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_0
            )

            self.preload_real_shift_matrix = preload_real_transforms_inverse(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_step
            )

            cube_in_1d = np.zeros(
                shape=(self.n_channels, self.grid.shape[0])
            )

            self.real_visibilities_from_cube(cube_in_1d.cube_in_1d)


#
#             self.preload_imag_transforms = transformer_util.preload_imag_transforms(
#                 grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths
#             )

    def real_visibilities_from_cube(self, cube_in_1d):

        if self.preload_transform:

            return real_visibilities_from_cube_via_preloaded_transform_and_shift_matrix(
                cube_1d=cube_in_1d,
                preloaded_reals=self.preload_real_transforms,
                shift_matrix_real=self.preload_real_shift_matrix,
                n_channels=self.n_channels
            )

        else:
            raise ValueError("Not implemented yet.")
            # return transformer_util.real_visibilities_jit(
            #     image_1d=image.in_1d_binned,
            #     grid_radians=self.grid,
            #     uv_wavelengths=self.uv_wavelengths,
            # )

#     def imag_visibilities_from_image(self, image):
#
#         if self.preload_transform:
#
#             return transformer_util.imag_visibilities_from_image_via_preload_jit(
#                 image_1d=image.in_1d_binned,
#                 preloaded_imags=self.preload_imag_transforms,
#             )
#
#         else:
#
#             return transformer_util.imag_visibilities_jit(
#                 image_1d=image.in_1d_binned,
#                 grid_radians=self.grid,
#                 uv_wavelengths=self.uv_wavelengths,
#             )
#
#     def visibilities_from_image(self, image):
#
#         real_visibilities = self.real_visibilities_from_image(image=image)
#         imag_visibilities = self.imag_visibilities_from_image(image=image)
#
#         return vis.Visibilities(
#             visibilities_1d=np.stack((real_visibilities, imag_visibilities), axis=-1)
#         )



if __name__ == "__main__":
    pass


    filename = "/Users/ccbh87/Desktop/Project/uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    #print(u_wavelengths.shape);exit()

    uv_wavelengths = np.dstack((u_wavelengths, v_wavelengths))
    #print(uv_wavelengths.shape);exit()

    x_grid_in_radians, y_grid_in_radians = grid_2d_in_radians(
        n_pixels=20, pixel_scale=0.1
    )
    #plot_grid(x_grid=x_grid_in_radians, y_grid=y_grid_in_radians)

    grid_1d_in_radians = np.array([
        np.ndarray.flatten(y_grid_in_radians),
        np.ndarray.flatten(x_grid_in_radians)
    ]).T

    transformer = Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid_1d_in_radians
    )
