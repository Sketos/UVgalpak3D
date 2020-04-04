import os
import sys

sys.path.insert(
    0,
    os.environ["Galpak3D"]
)
import galpak


class Model:
    def __init__(self, shape):

        self.galaxy_model = galpak.DiskModel(
            flux_profile='exponential', rotation_curve='isothermal'
        )

        self.shape=shape

    def create_cube(self, galaxy_parameters, dv):

        cube, _, _, _ = self.galaxy_model._create_cube(
            galaxy_parameters, self.shape, dv, galaxy_parameters.z
        )

        return cube


if __name__ == "__main__":

    # galaxy_model = galpak.DiskModel(
    #     flux_profile='exponential', rotation_curve='isothermal'
    # )

    model_obj = model()
    print(model_obj.galaxy_model)
