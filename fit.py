import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from visibilities import load_uv_wavelengths_from_fits, load_visibilities_from_fits
from grid import grid_2d_in_radians
from transformer import Transformer
from plot_utils import plot_cube, plot_visibilities

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool


# --- #



# --- #

sys.path.insert(
    0,
    os.environ["Galpak3D"]
)
import galpak

# ...
sys.path.append(
    os.environ["GitHub"] + "/utils"
)
from emcee_wrapper import emcee_wrapper


class Fit:
    def __init__(self, transformer, model):

        self.transformer = transformer
        self.model



    def model_cube(self, galaxy_model, galaxy_parameters, dv):

        cube, _, _, _ = galaxy_model._create_cube(
            galaxy_parameters, self.transformer.cube_shape, dv, galaxy_parameters.z
        )

        return cube




if __name__ == "__main__":
    pass

    # NOTE: are these
    filename = "./uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    #print(u_wavelengths.shape);exit()

    uv_wavelengths = np.stack((u_wavelengths, v_wavelengths), axis=-1)
    #print(uv_wavelengths.shape);exit()


    filename = "./visibilities.fits"
    visibilities = fits.getdata(filename=filename)

    # real_visibilities, imag_visibilities = load_visibilities_from_fits(
    #     filename=filename
    # )
    # exit()
    # visibilities = np.stack((real_visibilities, imag_visibilities), axis=-1)

    #plot_visibilities(visibilities=visibilities);exit()
    if visibilities.shape == uv_wavelengths.shape:
        print("OK LETS GO")

    #print(visibilities.shape)
    #exit()
    #visibilities = np.random.normal(0.0, 1.0, size=uv_wavelengths.shape)

    n_pixels = 50
    pixel_scale = 0.1
    x_grid_in_radians, y_grid_in_radians = grid_2d_in_radians(
        n_pixels=n_pixels, pixel_scale=pixel_scale
    )
    #plot_grid(x_grid=x_grid_in_radians, y_grid=y_grid_in_radians)

    grid_1d_in_radians = np.array([
        np.ndarray.flatten(y_grid_in_radians),
        np.ndarray.flatten(x_grid_in_radians)
    ]).T

    transformer = Transformer(
        uv_wavelengths=uv_wavelengths, grid=grid_1d_in_radians, preload_transform=True
    )

    # plt.figure()
    # plt.imshow(transformer.preload_real_shift_matrix, aspect="auto")
    # plt.show()
    # exit()

    # galaxy_parameters = galpak.GalaxyParameters()
    # galaxy_parameters.x = int(n_pixels / 2.0)
    # galaxy_parameters.y = int(n_pixels / 2.0)
    # galaxy_parameters.z = transformer.n_channels / 2.0
    # galaxy_parameters.flux = 0.25
    # radius = 1.0 # in units of arcsec
    # galaxy_parameters.radius = radius / pixel_scale
    # galaxy_parameters.inclination = 50.0
    # galaxy_parameters.pa = 65.0
    # turnover_radius = 0.2 # in units of arcsec
    # galaxy_parameters.turnover_radius = turnover_radius / pixel_scale
    # galaxy_parameters.maximum_velocity = 300.0
    # galaxy_parameters.velocity_dispersion = 50.0
    #
    # #print(galaxy_parameters);exit()
    # # print(dir(galaxy_parameters))
    # # #print(galaxy_parameters.from_ndarray(a))
    # # galaxy_parameters = galpak.GalaxyParameters.from_ndarray(a=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # # print(galaxy_parameters)

    fit = Fit(transformer=transformer)

    def model(theta):
        print("IM AM BEING CALLED: MODEL")
        galaxy_model = galpak.DiskModel(
            flux_profile='exponential', rotation_curve='isothermal'
        )
        dv = 50.0

        galaxy_parameters = galpak.GalaxyParameters.from_ndarray(theta)

        model_cube = fit.model_cube(
            galaxy_model=galaxy_model, galaxy_parameters=galaxy_parameters, dv=dv
        )
        model_cube = model_cube.data
        #model(theta)

        model_cube_reshaped = model_cube.reshape(
            model_cube.shape[0], int(model_cube.shape[1] * model_cube.shape[2])
        )

        y_model = transformer.visibilities_from_cube(
            cube_in_1d=model_cube_reshaped
        )
        #print(y_model.shape)

        return y_model




    def log_likelihood_helper(obj, theta):
        print("IM AM BEING CALLED: log_likelihood_helper")
        y_model = model(theta)

        print("SHAPES = ", y_model.shape, obj.y.shape)

        # NOTE: replace self with name of the object
        chi_squared_real = np.sum(
            (obj.y[:, :, 0] - y_model[:, :, 0])**2.0 / obj.yerr[:, :, 0]**2.0
        )

        chi_squared_imag = np.sum(
            (obj.y[:, :, 1] - y_model[:, :, 1])**2.0 / obj.yerr[:, :, 1]**2.0
        )

        return -0.5 * (chi_squared_real + chi_squared_imag)



    class emcee_wrapper:

        def __init__(self, x, y, yerr, mcmc_limits, n_walkers=500):

            self.x = x
            self.y = y

            # NOTE: What do I do in the case where I dont have yerr available
            if yerr is None:
                self.yerr = np.ones(shape=self.y.shape)
            else:
                self.yerr = yerr

            # ...
            self.n_walkers = n_walkers

            # ...
            if mcmc_limits is None:
                raise ValueError
            else:
                self.theta = np.zeros(
                    shape=mcmc_limits.shape[0]
                )

            self.n_dim = len(self.theta)

            self.par_min, self.par_max = mcmc_limits.T
            self.par = self.initialize(
                par_min=self.par_min,
                par_max=self.par_max,
                n_dim=self.n_dim,
                n_walkers=n_walkers
            )

        @staticmethod
        def initialize(par_min, par_max, n_dim, n_walkers):

            return np.array([
                par_min + (par_max - par_min) * np.random.rand(n_dim)
                for i in range(n_walkers)
            ])


        def log_prior(self, theta):

            # NOTE: make this a function
            condition = np.zeros(
                shape=self.n_dim, dtype=bool
            )
            for n in range(len(theta)):
                if self.par_min[n] < theta[n] < self.par_max[n]:
                    condition[n] = True

            if np.all(condition):
                return 0.0
            else:
                return -np.inf


        def log_likelihood(self, theta):

            # y_model = model(self.x, theta)
            #
            # return -0.5 * np.sum(
            #     (self.y - y_model)**2.0 / self.yerr**2.0 + np.log(2.0 * np.pi * self.yerr**2.0)
            # )

            print("theta = ", theta)
            _log_likelihood = log_likelihood_helper(self, theta)
            print("_log_likelihood = ", _log_likelihood)
            return _log_likelihood


        def log_probability(self,theta):

            lp = self.log_prior(theta)

            if not np.isfinite(lp):
                return -np.inf
            else:
                return lp + self.log_likelihood(theta=theta)


        # NOTE: This function makes the "log_probability" pickleable.
        def __call__(self, theta):
            return self.log_probability(theta)


        def run(self, parallel=False):

            if parallel:
                with Pool() as pool:
                    sampler = emcee.EnsembleSampler(
                        self.n_walkers, self.n_dim, self.log_probability, pool=pool
                    )
                    sampler.run_mcmc(
                        self.par, self.n_walkers, progress=True
                    )
            else:
                sampler = emcee.EnsembleSampler(
                    self.n_walkers, self.n_dim, self.log_probability
                )
                sampler.run_mcmc(
                    self.par, self.n_walkers, progress=True
                )

            return sampler


    # model_parameter_limits = np.array([
    #     [5.0,15.0], [5.0,15.0], [10.0,20.0], [0.0,1.0], [0.0,20.0], [25.0,75.0], [60.0,70.0], [1.0,3.0], [250.0,350.0], [40.0,60.0],
    # ])
    model_parameter_limits = np.array([
        [20.0,30.0], [20.0,30.0], [10.0,20.0], [0.0,1.0], [0.0,20.0], [40.0,60.0], [60.0,70.0], [1.0,3.0], [250.0,350.0], [40.0,60.0],
    ])

    obj = emcee_wrapper(
        x=uv_wavelengths, y=visibilities, yerr=None, mcmc_limits=model_parameter_limits, n_walkers=200
    )


    sampler = obj.run(parallel=False)

    flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)

    # plt.figure()
    # plt.errorbar(x, y, yerr=yerr, linestyle="None", marker="o")
    # plt.show()
    # exit()

    import corner

    fig = corner.corner(
        flat_samples
    )
    plt.show()

    exit()


    """
    galaxy_model = galpak.DiskModel(
        flux_profile='exponential', rotation_curve='isothermal'
    )

    dv = 50.0
    model_cube = fit.model_cube(
        galaxy_model=galaxy_model, galaxy_parameters=galaxy_parameters, dv=dv
    )
    model_cube = model_cube.data




    #print(model_cube.data.shape);exit()

    model_cube_reshaped = model_cube.reshape(
        model_cube.shape[0], int(model_cube.shape[1] * model_cube.shape[2])
    )
    #print(model_cube_reshaped.shape);exit()
    # plot_cube(model_cube, ncols=6)
    # exit()

    # plt.figure()
    # plt.imshow(model_cube_reshaped, aspect="auto")
    # plt.show()
    # exit()


    # #n = int(model_cube_reshaped.shape[0]/2.0)
    # n = 0
    # real_visibilities = transformer.real_visibilities_from_image(
    #     image_in_1d=model_cube_reshaped[n, :],
    #     preloaded_reals=transformer.preload_real_transforms
    # )
    # imag_visibilities = transformer.imag_visibilities_from_image(
    #     image_in_1d=model_cube_reshaped[n, :],
    #     preloaded_imags=transformer.preload_imag_transforms
    # )
    # plot_visibilities(visibilities=np.stack((real_visibilities, imag_visibilities), axis=-1))
    # exit()


    # visibilities = transformer.visibilities_from_cube(
    #     cube_in_1d=model_cube_reshaped
    # )
    # #print(visibilities.shape)





    visibilities = transformer.visibilities_from_cube(
        cube_in_1d=model_cube_reshaped
    )
    plot_visibilities(visibilities=visibilities)


    exit()
    """
