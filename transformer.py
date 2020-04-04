import numpy as np
import matplotlib.pyplot as plt
import time

from visibilities import load_uv_wavelengths_from_fits
from grid import grid_2d_in_radians
from mask import z_mask

# NOTE: only keep packages that are needed.
from transformer_utils import *
from plot_utils import *


class Transformer:
    def __init__(self, uv_wavelengths, grid=None, z_mask=None, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths
        #print(self.uv_wavelengths.shape)#; exit()

        # if self.uv_wavelengths.shape[0] == len(z_mask):
        #     self.uv_wavelengths = self.uv_wavelengths[z_mask]
        # else:
        #     raise ValueError


        exit()

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

        # TODO: Make the grig a class
        # NOTE: At this point the grid is in radians. This will be made into a function that will hold both grid and grid.in_radians
        self.grid = grid

        self.total_pixels = self.grid.shape[0]

        #self.grid_shape_in_2d = (int(np.sqrt(self.grid.shape[0])), int(np.sqrt(self.grid.shape[0])), self.grid.shape[-1])

        self.cube_shape = (
            self.n_channels,
            int(np.sqrt(self.grid.shape[0])),
            int(np.sqrt(self.grid.shape[0]))
        )




        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = preload_real_transforms(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_0
            )

            self.preload_real_shift_matrix = preload_real_transforms_inverse(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_step
            )

            self.preload_imag_transforms = preload_imag_transforms(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_0
            )

            self.preload_imag_shift_matrix = preload_imag_transforms_inverse(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths_step
            )

    # NOTE: This is probably not needed. It is just for testing
    def real_visibilities_from_image(self, image_in_1d, preloaded_reals):

        return real_visibilities_from_image_via_preloaded_transform(
            image_1d=image_in_1d,
            preloaded_reals=preloaded_reals,
        )

    # NOTE: This is probably not needed. It is just for testing
    def imag_visibilities_from_image(self, image_in_1d, preloaded_imags):

        return imag_visibilities_from_image_via_preloaded_transform(
            image_1d=image_in_1d,
            preloaded_imags=preloaded_imags,
        )

    """ # NOTE: I am not sure if it's worth calculating them separetly
    def real_visibilities_from_cube(self, cube_in_1d):

        if self.preload_transform:

            return real_visibilities_from_cube_via_preloaded_transform_and_shift_matrix(
                cube_1d=cube_in_1d,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
                shift_matrix_real=self.preload_real_shift_matrix,
                shift_matrix_imag=self.preload_imag_shift_matrix
            )

        else:
            raise ValueError("Not implemented yet.")
            # return transformer_util.real_visibilities_jit(
            #     image_1d=image.in_1d_binned,
            #     grid_radians=self.grid,
            #     uv_wavelengths=self.uv_wavelengths,
            # )

    def imag_visibilities_from_cube(self, cube_in_1d):

        if self.preload_transform:

            return imag_visibilities_from_cube_via_preloaded_transform_and_shift_matrix(
                cube_1d=cube_in_1d,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
                shift_matrix_real=self.preload_real_shift_matrix,
                shift_matrix_imag=self.preload_imag_shift_matrix
            )

        else:
            raise ValueError("Not implemented yet.")
            # return transformer_util.imag_visibilities_jit(
            #     image_1d=image.in_1d_binned,
            #     grid_radians=self.grid,
            #     uv_wavelengths=self.uv_wavelengths,
            # )
    """

    def visibilities_from_cube(self, cube_in_1d):

        if self.preload_transform:
            start = time.time()
            real_visibilities, imag_visibilities = visibilities_from_cube_via_preloaded_transform_and_shift_matrix(
                cube_1d=cube_in_1d,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
                shift_matrix_real=self.preload_real_shift_matrix,
                shift_matrix_imag=self.preload_imag_shift_matrix
            )
            end = time.time()
            print("time for dft:", end - start)
        else:
            raise ValueError

        return np.stack((real_visibilities, imag_visibilities), axis=-1)



if __name__ == "__main__":

    filename = "/Users/ccbh87/Desktop/Project/uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    #print(u_wavelengths.shape);exit()

    uv_wavelengths = np.dstack((u_wavelengths, v_wavelengths))
    #plot_uv_wavelengths(uv_wavelengths=uv_wavelengths);exit()
    #print(uv_wavelengths.shape);exit()

    n_pixels=50
    pixel_scale=0.1
    x_grid_in_radians, y_grid_in_radians = grid_2d_in_radians(
        n_pixels=n_pixels, pixel_scale=pixel_scale
    )
    # plot_grid(x_grid=x_grid_in_radians, y_grid=y_grid_in_radians)
    # exit()

    grid_1d_in_radians = np.array([
        np.ndarray.flatten(y_grid_in_radians),
        np.ndarray.flatten(x_grid_in_radians)
    ]).T
    #print(grid_1d_in_radians.shape);exit()

    #_z_mask = z_mask(n=uv_wavelengths.shape[0], nmin=8, nmax=24)

    transformer = Transformer(
        uv_wavelengths=uv_wavelengths,
        grid=grid_1d_in_radians,
        z_mask=z_mask(
            n=uv_wavelengths.shape[0], zmin=8, zmax=24
        ),
        preload_transform=True
    )

    cube_in_1d = np.random.normal(
        0.0, 1.0, size=(transformer.n_channels, transformer.total_pixels)
    )
    visibilities = transformer.visibilities_from_cube(cube_in_1d=cube_in_1d)
    print(visibilities.shape)

    exit()



    # image_1d = np.random.normal(0.0, 10.0, size=grid_1d_in_radians.shape[0])
    #
    # start = time.time()
    # c = 5
    # _real_visibilities = real_visibilities_from_image(
    #     image_1d=image_1d, grid_radians=grid_1d_in_radians, uv_wavelengths=uv_wavelengths[c]
    # )
    # _imag_visibilities = imag_visibilities_from_image(
    #     image_1d=image_1d, grid_radians=grid_1d_in_radians, uv_wavelengths=uv_wavelengths[c]
    # )
    # end = time.time()
    # print(end-start)


    """# NOTE: OK
    start = time.time()
    visibilities = compute_visibilities_from_image_slow(
        u=uv_wavelengths[c, :, 0],
        v=uv_wavelengths[c, :, 1],
        n_pixels=n_pixels,
        x_rad=x_grid_in_radians,
        y_rad=y_grid_in_radians,
        skymodel=image_1d.reshape(n_pixels, n_pixels)
    )
    end = time.time()
    print(end-start)

    plt.figure()
    plt.plot(
        _real_visibilities,
        _imag_visibilities,
        linestyle="None",
        marker="o",
        markersize=5,
        color="b",
        alpha=0.25
    )
    plt.plot(
        visibilities.real,
        visibilities.imag,
        linestyle="None",
        marker="o",
        markersize=2,
        color="r",
        alpha=0.95
    )
    plt.show()
    exit()
    """


    """
    transformer = Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid_1d_in_radians
    )

    # shift_matrix = transformer.preload_real_shift_matrix + 1j * transformer.preload_imag_shift_matrix
    #
    # n = 5
    # shift_matrix_real, shift_matrix_imag = shift_matrix_to_power_of_n(
    #     n=n,
    #     shift_matrix_real=transformer.preload_real_shift_matrix,
    #     shift_matrix_imag=transformer.preload_imag_shift_matrix
    # )
    #
    # shift_matrix = shift_matrix**n

    # print(shift_matrix_real.shape)
    # print(shift_matrix.real.shape)
    # exit()

    # plt.figure()
    # plt.imshow(shift_matrix_real, aspect="auto")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(shift_matrix.real, aspect="auto")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(shift_matrix.real-shift_matrix_real, aspect="auto")
    # plt.colorbar()
    #
    # plt.show()
    # exit()

    # print(np.allclose(shift_matrix.real, shift_matrix_real))
    # print(np.allclose(shift_matrix.imag, shift_matrix_imag))
    #
    # exit()


    cube_in_1d = np.ones(
        shape=(uv_wavelengths.shape[0], grid_1d_in_radians.shape[0])
    )
    for i in range(cube_in_1d.shape[0]):
        cube_in_1d[i] = image_1d

    start = time.time()
    real_visibilities, imag_visibilities, shift_matrix_real_temp, shift_matrix_imag_temp = transformer.real_visibilities_from_cube(cube_in_1d=cube_in_1d)
    # #imag_visibilities = transformer.imag_visibilities_from_cube(cube_in_1d=cube_in_1d)
    end = time.time()
    print(end-start)

    #print(cube_in_1d.shape, transformer.preload_real_transforms.shape, transformer.preload_real_shift_matrix.shape);exit()

    # start = time.time()
    # visibilities = visibilities_from_cube_via_preloaded_transform_and_shift_matrix(
    #     cube_1d=cube_in_1d,
    #     preloaded=transformer.preload_real_transforms + 1j * transformer.preload_imag_transforms,
    #     shift_matrix=transformer.preload_real_shift_matrix + 1j * transformer.preload_imag_shift_matrix
    # )
    # end = time.time()
    # print(end-start)


    # NOTE: The shift matrices are fine ...
    # shift_matrix = transformer.preload_real_shift_matrix + 1j * transformer.preload_imag_shift_matrix
    # shift_matrix = shift_matrix**cube_in_1d.shape[0]
    # print(np.allclose(shift_matrix.real, shift_matrix_real_temp))
    # print(np.allclose(shift_matrix.imag, shift_matrix_imag_temp))
    # exit()

    plt.figure()
    plt.plot(
        _real_visibilities,
        _imag_visibilities,
        linestyle="None",
        marker="o",
        markersize=5,
        color="b",
        alpha=0.25
    )
    plt.plot(
        real_visibilities[c],
        imag_visibilities[c],
        linestyle="None",
        marker="o",
        markersize=2,
        color="r",
        alpha=0.95
    )

    # plt.plot(
    #     _real_visibilities,
    #     visibilities[c].real,
    #     linestyle="None",
    #     marker="o",
    #     markersize=5,
    #     color="b",
    #     alpha=0.25
    # )
    # xmin = -50.0
    # xmax = 50.0
    # x = np.linspace(xmin, xmax, 100)
    # plt.plot(x, x, color="black")

    plt.show()
    """


    """ # NOTE: OK
    cube_in_1d = np.ones(
        shape=(uv_wavelengths.shape[0], grid_1d_in_radians.shape[0])
    )
    for i in range(cube_in_1d.shape[0]):
        cube_in_1d[i] = image_1d

    visibilities = compute_visibilities_from_cube_slow(
        n_channels=uv_wavelengths.shape[0],
        u=uv_wavelengths[:, :, 0],
        v=uv_wavelengths[:, :, 1],
        skymodel=cube_in_1d.reshape(uv_wavelengths.shape[0], n_pixels, n_pixels),
        n_pixels=n_pixels,
        x_rad=x_grid_in_radians,
        y_rad=y_grid_in_radians
    )

    plt.figure()
    plt.plot(
        _real_visibilities,
        _imag_visibilities,
        linestyle="None",
        marker="o",
        markersize=5,
        color="b",
        alpha=0.25
    )
    plt.plot(
        visibilities.real[c],
        visibilities.imag[c],
        linestyle="None",
        marker="o",
        markersize=2,
        color="r",
        alpha=0.95
    )
    plt.show()
    """
















    # # NOTE : ...
    # import os
    # import sys
    # if os.environ["HOME"] == "/Users/ccbh87":
    #     COSMA_HOME = os.environ["COSMA_HOME_local"]
    #     COSMA_DATA = os.environ["COSMA7_DATA_local"]
    # elif os.environ["HOME"] == "/cosma/home/durham/dc-amvr1":
    #     COSMA_HOME = os.environ["COSMA_HOME_host"]
    #     COSMA_DATA = os.environ["COSMA7_DATA_host"]
    #
    # # ...
    # workspace_HOME_path = COSMA_HOME + "/workspace"
    # workspace_DATA_path = COSMA_DATA + "/workspace"
    #
    # # ...
    # autolens_version = "0.40.0"
    #
    # # ...
    # import autofit as af
    # af.conf.instance = af.conf.Config(
    #     config_path=workspace_DATA_path + "/config" + "_" + autolens_version,
    #     output_path=workspace_DATA_path + "/output")
    # import autolens as al
    #
    # grid = al.grid.uniform(
    #     shape_2d=(
    #         n_pixels,
    #         n_pixels
    #     ),
    #     pixel_scales=(
    #         pixel_scale,
    #         pixel_scale
    #     ),
    #     sub_size=1
    # )
    #
    # transformer = al.transformer(
    #     uv_wavelengths=uv_wavelengths[c], grid_radians=grid.in_radians, preload_transform=False
    # )
    # image = al.array.ones(shape_2d=grid.shape_2d, pixel_scales=grid.pixel_scales)
    # real_visibilities_al = transformer.real_visibilities_from_image(image=image)
    # imag_visibilities_al = transformer.imag_visibilities_from_image(image=image)
    #
    # #print(np.allclose(real_visibilities, real_visibilities_al))
    # #print(np.allclose(imag_visibilities, imag_visibilities_al))
    # #print(real_visibilities_al)
    # #print(real_visibilities)
    # #plt.show()
