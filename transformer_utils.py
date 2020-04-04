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
def real_visibilities_from_image_via_preloaded_transform(image_1d, preloaded_reals):

    real_visibilities = np.zeros(
        shape=(preloaded_reals.shape[1])
    )

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(preloaded_reals.shape[1]):
            real_visibilities[vis_1d_index] += (
                image_1d[image_1d_index] * preloaded_reals[image_1d_index, vis_1d_index]
            )

    return real_visibilities


@decorator_util.jit()
def real_visibilities_from_image(image_1d, grid_radians, uv_wavelengths):

    real_visibilities = np.zeros(
        shape=(uv_wavelengths.shape[0])
    )

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            real_visibilities[vis_1d_index] += image_1d[image_1d_index] * np.cos(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return real_visibilities


@decorator_util.jit()
def imag_visibilities_from_image(image_1d, grid_radians, uv_wavelengths):

    imag_visibilities = np.zeros(
        shape=(uv_wavelengths.shape[0])
    )

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            imag_visibilities[vis_1d_index] += image_1d[image_1d_index] * np.sin(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return imag_visibilities


@decorator_util.jit()
def imag_visibilities_from_image_via_preloaded_transform(image_1d, preloaded_imags):

    imag_visibilities = np.zeros(
        shape=(preloaded_imags.shape[1])
    )

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(preloaded_imags.shape[1]):
            imag_visibilities[vis_1d_index] += (
                image_1d[image_1d_index] * preloaded_imags[image_1d_index, vis_1d_index]
            )

    return imag_visibilities




@decorator_util.jit()
def visibilities_from_cube_via_preloaded_transform_and_shift_matrix(cube_1d, preloaded_reals, preloaded_imags, shift_matrix_real, shift_matrix_imag):

    real_visibilities = np.zeros(
        shape=(cube_1d.shape[0], preloaded_reals.shape[1])
    )
    imag_visibilities = np.zeros(
        shape=(cube_1d.shape[0], preloaded_reals.shape[1])
    )

    for image_1d_index in range(cube_1d.shape[1]):
        for vis_1d_index in range(preloaded_reals.shape[1]):
            real_visibilities[0, vis_1d_index] += (
                cube_1d[0, image_1d_index]
                * preloaded_reals[image_1d_index, vis_1d_index]
            )

    shift_matrix_real_temp = shift_matrix_real
    shift_matrix_imag_temp = shift_matrix_imag

    shift_matrix_real_holder = shift_matrix_real
    shift_matrix_imag_holder = shift_matrix_imag

    for n in range(1, cube_1d.shape[0]):

        matrix_real_temp = np.zeros(shape=shift_matrix_real.shape)
        matrix_imag_temp = np.zeros(shape=shift_matrix_imag.shape)

        for image_1d_index in range(preloaded_reals.shape[0]):
            for vis_1d_index in range(preloaded_reals.shape[1]):

                matrix_real_temp[image_1d_index, vis_1d_index] += (
                    preloaded_reals[image_1d_index, vis_1d_index] * shift_matrix_real_temp[image_1d_index, vis_1d_index]
                    - preloaded_imags[image_1d_index, vis_1d_index] * shift_matrix_imag_temp[image_1d_index, vis_1d_index]
                )

                matrix_imag_temp[image_1d_index, vis_1d_index] += (
                    preloaded_reals[image_1d_index, vis_1d_index] * shift_matrix_imag_temp[image_1d_index, vis_1d_index]
                    + preloaded_imags[image_1d_index, vis_1d_index] * shift_matrix_real_temp[image_1d_index, vis_1d_index]
                )

        for image_1d_index in range(preloaded_reals.shape[0]):
            for vis_1d_index in range(preloaded_reals.shape[1]):

                real_visibilities[n, vis_1d_index] += (
                    cube_1d[n, image_1d_index]
                    * matrix_real_temp[image_1d_index, vis_1d_index]
                )

                imag_visibilities[n, vis_1d_index] += (
                    cube_1d[n, image_1d_index]
                    * matrix_imag_temp[image_1d_index, vis_1d_index]
                )

        shift_matrix_real_temp = np.zeros(shape=shift_matrix_real.shape)
        shift_matrix_imag_temp = np.zeros(shape=shift_matrix_imag.shape)

        for image_1d_index in range(preloaded_reals.shape[0]):
            for vis_1d_index in range(preloaded_reals.shape[1]):

                shift_matrix_real_temp[image_1d_index, vis_1d_index] += (
                    shift_matrix_real_holder[image_1d_index, vis_1d_index] * shift_matrix_real[image_1d_index, vis_1d_index]
                    - shift_matrix_imag_holder[image_1d_index, vis_1d_index] * shift_matrix_imag[image_1d_index, vis_1d_index]
                )

                shift_matrix_imag_temp[image_1d_index, vis_1d_index] += (
                    shift_matrix_real_holder[image_1d_index, vis_1d_index] * shift_matrix_imag[image_1d_index, vis_1d_index]
                    + shift_matrix_imag_holder[image_1d_index, vis_1d_index] * shift_matrix_real[image_1d_index, vis_1d_index]
                )

        shift_matrix_real_holder = shift_matrix_real_temp
        shift_matrix_imag_holder = shift_matrix_imag_temp


    return real_visibilities, imag_visibilities
