import numpy as np
import matplotlib.pyplot as plt

from visibilities import load_uv_wavelengths_from_fits
from grid import grid_2d_in_radians

#from transformer_utils import *
from plot_utils import *









# a = 2.232
# b = 3.452
# z = a + 1j * b
#
# def power_of_complex_number(z, n):
#
#     z_real = z.real
#     z_imag = z.imag
#
#     z_real_temp = z_real
#     z_imag_temp = z_imag
#     z_real_holder = z_real
#     z_imag_holder = z_imag
#
#     for i in range(n - 1):
#         z_real_temp = z_real_holder * z_real - z_imag_holder * z_imag
#         z_imag_temp = z_real_holder * z_imag + z_imag_holder * z_real
#         z_real_holder = z_real_temp
#         z_imag_holder = z_imag_temp
#
#     return z_real_temp + 1j * z_imag_temp
#
# n = 10
# print(z**n)
# print(power_of_complex_number(z, n))


z_matrix = np.array([
    [2.0 + 3.0j, 5.0 - 4.0j, 4.5 + 2.0j],
    [1.0 + 3.0j, 1.0 + 4.0j, 4.5 - 2.0j]
])

def power_of_complex_matrix(z_matrix, n):

    z_matrix_real = z_matrix.real
    z_matrix_imag = z_matrix.imag

    z_matrix_real_temp = z_matrix.real
    z_matrix_imag_temp = z_matrix.imag

    z_matrix_real_holder = z_matrix.real
    z_matrix_imag_holder = z_matrix.imag

    for k in range(n - 1):

        z_matrix_real_temp = np.zeros(shape=z_matrix_real.shape)
        z_matrix_imag_temp = np.zeros(shape=z_matrix_imag.shape)

        for i in range(z_matrix.shape[0]):
            for j in range(z_matrix.shape[1]):

                z_matrix_real_temp[i, j] += (
                    z_matrix_real_holder[i, j] * z_matrix_real[i, j] - z_matrix_imag_holder[i, j] * z_matrix_imag[i, j]
                )

                z_matrix_imag_temp[i, j] += (
                    z_matrix_real_holder[i, j] * z_matrix_imag[i, j] + z_matrix_imag_holder[i, j] * z_matrix_real[i, j]
                )

        z_matrix_real_holder = z_matrix_real_temp
        z_matrix_imag_holder = z_matrix_imag_temp

    return z_matrix_real_temp + 1j * z_matrix_imag_temp

n = 3
_z_matrix = power_of_complex_matrix(z_matrix, n)
z_matrix = z_matrix**n

print(np.allclose(_z_matrix.real, z_matrix.real))
print(np.allclose(_z_matrix.imag, z_matrix.imag))

exit()


"""
def _compute_visibilities_from_cube(n_channels, u, v, skymodel, n_pixels, x_rad, y_rad):
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


def _compute_visibilities_from_image(u, v, n_pixels, x_rad, y_rad, skymodel):
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


if __name__ == "__main__":

    filename = "/Users/ccbh87/Desktop/Project/uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    #print(u_wavelengths.shape);exit()

    uv_wavelengths = np.dstack((u_wavelengths, v_wavelengths))
    #plot_uv_wavelengths(uv_wavelengths=uv_wavelengths)
    #print(uv_wavelengths.shape);exit()

    n_pixels=20
    pixel_scale=0.1
    x_grid_in_radians, y_grid_in_radians = grid_2d_in_radians(
        n_pixels=20, pixel_scale=0.1
    )

    image = np.random.normal(0.0, 1.0, size=(n_pixels,n_pixels))
    cube = np.zeros(shape=(uv_wavelengths.shape[0], n_pixels, n_pixels))
    for i in range(cube.shape[0]):
        cube[i, :, :] = image[:, :]


    visibilities_from_cube = _compute_visibilities_from_cube(
        n_channels=uv_wavelengths.shape[0],
        u=uv_wavelengths[:, :, 0],
        v=uv_wavelengths[:, :, 1],
        skymodel=cube,
        n_pixels=n_pixels,
        x_rad=x_grid_in_radians,
        y_rad=y_grid_in_radians
    )

    c = 1
    visibilities_from_imag = _compute_visibilities_from_image(
        u=uv_wavelengths[c, :, 0],
        v=uv_wavelengths[c, :, 1],
        n_pixels=n_pixels,
        x_rad=x_grid_in_radians,
        y_rad=y_grid_in_radians,
        skymodel=image
    )

    plt.figure()
    plt.plot(
        visibilities_from_imag.real,
        visibilities_from_imag.imag,
        linestyle="None",
        marker="o",
        markersize=5,
        color="b",
        alpha=0.25
    )
    plt.plot(
        visibilities_from_cube.real[c],
        visibilities_from_cube.imag[c],
        linestyle="None",
        marker="o",
        markersize=2,
        color="r",
        alpha=0.95
    )

    # plt.figure()
    # plt.plot(
    #     visibilities_from_imag.real - visibilities_from_cube.real[c],
    #     visibilities_from_imag.imag - visibilities_from_cube.imag[c],
    #     linestyle="None",
    #     marker="o",
    #     markersize=2,
    #     color="r",
    #     alpha=0.95
    # )
    plt.show()

"""
