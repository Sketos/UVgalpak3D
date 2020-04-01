import numpy as np

import decorator_util as decorator_util


@decorator_util.jit()
def preload_real_transforms(grid_radians, uv_wavelengths):

    preloaded_real_transforms = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_real_transforms[image_1d_index, vis_1d_index] += np.cos(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_real_transforms


@decorator_util.jit()
def preload_real_transforms_inverse(grid_radians, uv_wavelengths):

    preloaded_real_transforms_inverse = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_real_transforms_inverse[image_1d_index, vis_1d_index] += np.cos(
                2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_real_transforms_inverse


@decorator_util.jit()
def preload_imag_transforms(grid_radians, uv_wavelengths):

    preloaded_imag_transforms = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_imag_transforms[image_1d_index, vis_1d_index] += np.sin(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_imag_transforms


@decorator_util.jit()
def preload_imag_transforms_inverse(grid_radians, uv_wavelengths):

    preloaded_imag_transforms_inverse = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_imag_transforms_inverse[image_1d_index, vis_1d_index] += np.sin(
                2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_imag_transforms_inverse


@decorator_util.jit()
def real_visibilities_from_cube_via_preloaded_transform_and_shift_matrix(cube_1d, preloaded_reals, shift_matrix_real, n_channels):

    real_visibilities = np.zeros(
        shape=(cube_1d.shape[0], preloaded_reals.shape[1])
    )

    for n in range(cube_1d.shape[0]):
        for image_1d_index in range(cube_1d.shape[1]):
            for vis_1d_index in range(preloaded_reals.shape[1]):
                real_visibilities[n, vis_1d_index] += (
                    cube_1d[n, image_1d_index] * preloaded_reals[image_1d_index, vis_1d_index] * shift_matrix_real[image_1d_index, vis_1d_index]**n
                )

    return real_visibilities


@decorator_util.jit()
def imag_visibilities_from_cube_via_preloaded_transform_and_shift_matrix(cube_1d, preloaded_imags, shift_matrix_imag, n_channels):

    imag_visibilities = np.zeros(
        shape=(cube_1d.shape[0], preloaded_imags.shape[1])
    )

    for n in range(cube_1d.shape[0]):
        for image_1d_index in range(cube_1d.shape[1]):
            for vis_1d_index in range(preloaded_imags.shape[1]):
                imag_visibilities[n, vis_1d_index] += (
                    cube_1d[n, image_1d_index] * preloaded_imags[image_1d_index, vis_1d_index] * shift_matrix_imag[image_1d_index, vis_1d_index]**n
                )

    return imag_visibilities
