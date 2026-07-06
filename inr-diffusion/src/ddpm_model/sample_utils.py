import matplotlib.pyplot as plt

import cbct_artifact_reduction.dataprocessing as dp


def plot_sample(name):
    numpy_array = dp.single_nifti_to_numpy(name)
    sample = numpy_array[0, 0, :, :]
    plt.imshow(sample)
    plt.show()
