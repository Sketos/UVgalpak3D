import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_grid(x_grid, y_grid):
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 4)
    )
    im0 = axes[0].imshow(x_grid, interpolation="None")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    figure.colorbar(im0, cax=cax, orientation='vertical')

    im1 = axes[1].imshow(y_grid, interpolation="None")
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    figure.colorbar(im1, cax=cax, orientation='vertical')

    plt.show()
