"""
@decorator_util.jit()
def shift_matrix_to_power_of_n(n, shift_matrix_real, shift_matrix_imag):

    shift_matrix_real_temp = shift_matrix_real
    shift_matrix_imag_temp = shift_matrix_imag

    shift_matrix_real_holder = shift_matrix_real
    shift_matrix_imag_holder = shift_matrix_imag

    for i in range(n - 1):

        shift_matrix_real_temp = np.zeros(shift_matrix_real.shape)
        shift_matrix_imag_temp = np.zeros(shift_matrix_imag.shape)

        for image_1d_index in range(shift_matrix_real.shape[0]):
            for vis_1d_index in range(shift_matrix_real.shape[1]):

                # z_real_temp = z_real_holder * z_real - z_imag_holder * z_imag
                # z_imag_temp = z_real_holder * z_imag + z_imag_holder * z_real
                # z_real_holder = z_real_temp
                # z_imag_holder = z_imag_temp

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


    return shift_matrix_real_temp, shift_matrix_imag_temp
"""

























###############################################################################
"""
# NOTE: THIS IS WORKING, but it's slower than the numba version
def visibilities_from_cube_via_preloaded_transform_and_shift_matrix(cube_1d, preloaded, shift_matrix):

    visibilities = np.zeros(
        shape=(
            cube_1d.shape[0],
            preloaded.shape[1]
        ),
        dtype="complex"
    )

    for n in range(cube_1d.shape[0]):
        for i in range(preloaded.shape[1]):
            visibilities[n, i] = np.sum(
                cube_1d[n, :]
                * (
                    preloaded[:, i] * (shift_matrix[:, i]**n)
                )
            )

    return visibilities


def compute_visibilities_from_cube_slow(n_channels, u, v, skymodel, n_pixels, x_rad, y_rad):
    visibilities_temp = np.zeros(
        shape=(
            n_channels,
            u.shape[-1]
        ),
        dtype="complex"
    )
    #du = u[0, 0] - u[1, 0]
    #dv = v[0, 0] - v[1, 0]
    du = u[0, :] - u[1, :]
    dv = v[0, :] - v[1, :]
    ft_components = np.zeros(
        shape=(
            u.shape[-1],
            n_pixels,
            n_pixels
        ),
        dtype="complex"
    )
    for i in range(u.shape[-1]):
        ft_components[i] = np.exp(
            -2j
            * np.pi
            * (
                x_rad
                * u[0, i]
                + y_rad
                * v[0, i]
            )
        )
    A = np.zeros(
        shape=(
            u.shape[-1],
            n_pixels,
            n_pixels
        ),
        dtype="complex"
    )
    for i in range(u.shape[-1]):
        A[i] = np.exp(
            2j
            * np.pi
            * (
                x_rad
                * du[i]
                + y_rad
                * dv[i]
            )
        )
    for n in range(n_channels):
        for i in range(u.shape[-1]):
            visibilities_temp[n, i] = np.sum(skymodel[n] * (ft_components[i] * (A[i]**n)))

    return visibilities_temp


def compute_visibilities_from_image_slow(u, v, n_pixels, x_rad, y_rad, skymodel):
    visibilities = np.zeros(
        shape=(
            u.shape[-1]
        ),
        dtype="complex"
    )
    ft_components = np.zeros(
        shape=(
            u.shape[-1],
            n_pixels,
            n_pixels
        ),
        dtype="complex"
    )
    for i in range(u.shape[-1]):
        ft_components[i] = np.exp(
            -2j
            * np.pi
            * (
                x_rad
                * u[i]
                + y_rad
                * v[i]
            )
        )
    for i in range(u.shape[-1]):
        visibilities[i] = np.sum(skymodel * ft_components[i])

    return visibilities
"""
