import numpy as np


def z_mask(n, zmin, zmax):

    mask = np.zeros(
        shape=int(n), dtype=int
    )

    mask[zmin:zmax] = 1

    return mask.astype(bool)


if __name__ == "__main__":

    array = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0],
    ])

# z_mask = np.zeros(shape=array.shape[0])
# z_mask = np.array([0, 1, 1, 1, 0, 1, 1])
#
# if array.shape[0] == len(z_mask):
#     pass
# else:
#     raise ValueError
#
#
# array_masked = array[z_mask.astype(bool)]
#
# #print(array_masked)
# print(array.shape)
# print(array_masked.shape)
