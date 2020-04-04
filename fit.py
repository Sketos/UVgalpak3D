import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from visibilities import load_uv_wavelengths_from_fits, load_visibilities_from_fits
from grid import grid_2d_in_radians
from transformer import Transformer
from model import Model
from plot_utils import plot_cube, plot_visibilities

import numpy as np
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool


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
    def __init__(self, transformer, galaxy_model, dv):

        self.transformer = transformer
        self.galaxy_model = galaxy_model

        self.dv = dv



    def model_cube(self, galaxy_model, galaxy_parameters):

        cube, _, _, _ = galaxy_model._create_cube(
            galaxy_parameters,
            shape=self.transformer.cube_shape,
            z_step_kms=self.dv,
            zo=galaxy_parameters.z
        )
        cube = cube.data

        return cube.reshape(
            self.transformer.n_channels, self.transformer.total_pixels
        )





def log_likelihood_helper(obj, theta):
    #print("IM AM BEING CALLED: log_likelihood_helper")

    y_model = model(theta)

    chi_squared_real = np.sum(
        (obj.y[:, :, 0] - y_model[:, :, 0])**2.0 / obj.yerr[:, :, 0]**2.0
    )
    chi_squared_imag = np.sum(
        (obj.y[:, :, 1] - y_model[:, :, 1])**2.0 / obj.yerr[:, :, 1]**2.0
    )

    return -0.5 * (chi_squared_real + chi_squared_imag)


class Data:
    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y

        if yerr is None:
            self.yerr = np.ones(shape=self.y.shape)
        else:
            self.yerr = yerr


if __name__ == "__main__":

    filename = "./uv_wavelengths.fits"
    u_wavelengths, v_wavelengths = load_uv_wavelengths_from_fits(
        filename=filename
    )
    uv_wavelengths = np.stack((u_wavelengths, v_wavelengths), axis=-1)

    filename = "./visibilities.fits"
    visibilities = fits.getdata(filename=filename)

    data = Data(
        x=uv_wavelengths,
        y=visibilities,
        yerr=None
    )

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
        uv_wavelengths=uv_wavelengths,
        grid=grid_1d_in_radians,
        preload_transform=True
    )

    galaxy_model = Model()

    fit = Fit(transformer=transformer, galaxy_model=galaxy_model.galaxy_model, dv=50.0)



    def model(theta):

        galaxy_parameters = galpak.GalaxyParameters.from_ndarray(theta)

        model_cube_reshaped = fit.model_cube(
            galaxy_model=fit.galaxy_model, galaxy_parameters=galaxy_parameters
        )

        model_visibilities = transformer.visibilities_from_cube(
            cube_in_1d=model_cube_reshaped
        )

        return model_visibilities


    # theta = [25.00, 25.00, 16.0, 2.50e-01, 7.50, 50.0, 65.0, 2.00, 300.00, 50.00]
    # model_visibilities = model(theta)
    #
    # residuals = data.y - model_visibilities
    # c = 16
    # real_residuals = residuals[c, :, 0]
    # imag_residuals = residuals[c, :, 1]
    # print(residuals.shape)
    # #log_likelihood_helper(obj, theta)
    # plt.figure()
    # plt.plot(np.ndarray.flatten(real_residuals), np.ndarray.flatten(imag_residuals), linestyle="None", marker=".")
    # plt.show()
    #
    # exit()




    class emcee_wrapper:

        def __init__(self, x, y, yerr, mcmc_limits, nwalkers=500, backend_filename="backend.h5"):

            self.x = x
            self.y = y

            # NOTE: What do I do in the case where I dont have yerr available
            if yerr is None:
                self.yerr = np.ones(shape=self.y.shape)
            else:
                self.yerr = yerr

            # ...
            if mcmc_limits is None:
                raise ValueError

            self.par_min = mcmc_limits[:, 0]
            self.par_max = mcmc_limits[:, 1]

            self.ndim = mcmc_limits.shape[0]

            self.nwalkers = nwalkers

            # NOTE: The backend is not working properly ...
            self.backend = emcee.backends.HDFBackend(
                filename=backend_filename
            )

            self.previous_nsteps = 0
            try:
                self.state = self.backend.get_last_sample()
            except:
                self.state = self.initialize_state(
                    par_min=self.par_min,
                    par_max=self.par_max,
                    ndim=self.ndim,
                    nwalkers=self.nwalkers
                )
                self.backend.reset(
                    self.nwalkers, self.ndim
                )

            self.previous_nsteps += self.backend.iteration


        @staticmethod
        def initialize_state(par_min, par_max, ndim, nwalkers):

            return np.array([
                par_min + (par_max - par_min) * np.random.rand(ndim)
                for i in range(nwalkers)
            ])


        def log_prior(self, theta):

            # NOTE: make this a function
            condition = np.zeros(
                shape=self.ndim, dtype=bool
            )
            for n in range(len(theta)):
                if self.par_min[n] < theta[n] < self.par_max[n]:
                    condition[n] = True

            if np.all(condition):
                return 0.0
            else:
                return -np.inf


        def log_likelihood(self, theta):

            # NOTE: pass the object so that the helper function has the data.
            _log_likelihood = log_likelihood_helper(
                self, theta
            )
            #print("_log_likelihood = ", _log_likelihood)

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


        def run(self, nsteps, parallel=False):

            if parallel:
                pool = Pool()
            else:
                pool = None

            sampler = emcee.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=self.ndim,
                log_prob_fn=self.log_probability,
                backend=self.backend,
                pool=pool
            )

            sampler.run_mcmc(
                initial_state=self.state,
                nsteps=nsteps - self.previous_nsteps,
                progress=True
            )

            return sampler


    # model_parameter_limits = np.array([
    #     [5.0,15.0], [5.0,15.0], [10.0,20.0], [0.0,1.0], [0.0,20.0], [25.0,75.0], [60.0,70.0], [1.0,3.0], [250.0,350.0], [40.0,60.0],
    # ])
    model_parameter_limits = np.array([
        [20.0,30.0], [20.0,30.0], [10.0,20.0], [0.0,1.0], [0.0,20.0], [40.0,60.0], [60.0,70.0], [1.0,3.0], [250.0,350.0], [40.0,60.0],
    ])

    obj = emcee_wrapper(
        x=uv_wavelengths, y=visibilities, yerr=None, mcmc_limits=model_parameter_limits, nwalkers=200
    )


    sampler = obj.run(nsteps=50, parallel=False)

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
