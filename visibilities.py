import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def load_visibilities_from_fits(filename):

    data = fits.getdata(filename=filename)

    # NOTE: The shape of the data array is (3, n_channels, n_visibilities), where n_visibilities is ...
    if len(data.shape) == 3:
        # NOTE:
        if data.shape[-1] == 1:
            raise ValueError
        elif data.shape[-1] == 2:
            real_visibilities, imag_visibilities = data
        elif data.shape[-1] == 3:
            raise ValueError
        else:
            raise ValueError

    return real_visibilities, imag_visibilities


def load_uv_wavelengths_from_fits(filename):

    # NOTE: check if file exists.

    data = fits.getdata(filename=filename)

    # NOTE: The shape of the data array is (3, n_channels, n_visibilities), where n_visibilities is ...
    if len(data.shape) == 3:
        # NOTE:
        if data.shape[0] == 1:
            raise ValueError
        elif data.shape[0] == 2:
            u_wavelengths, v_wavelengths = data
        elif data.shape[0] == 3:
            u_wavelengths, v_wavelengths, w_wavelengths = data
        else:
            raise ValueError

    # plt.figure()
    # plt.plot(
    #     u_wavelength[0],
    #     v_wavelength[0],
    #     linestyle="None",
    #     marker="."
    # )
    # plt.show()
    # exit()

    return u_wavelengths, v_wavelengths

def plot_uv_wavelengths():
    pass

def load_interferometric_data_from_fits(uv_wavelengths_filename):

    # NOTE:
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=uv_wavelengths_filename
    )


if __name__ =="__main__":

    filename = "/Users/ccbh87/Desktop/Project/uv_wavelengths.fits"
    load_uv_wavelengths_from_fits(filename=filename)
