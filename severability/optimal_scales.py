"""Detect optimal scales from a scale scan."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.signal import find_peaks


def _pool2d_rand(A, kernel_size, stride, padding=0):
    """Computes 2D average-pooling.

    Average-pooling ignores padded values and diagonal values.

    Args:
        A (array): input 2D array
        kernel_size (int): size of the window over which we take pool
        stride (int): stride of the window
        padding (int): implicit NAN paddings on both sides of the input

    Returns:
        Average-pooled 2D array
    """
    # Padding with NAN
    A = np.pad(A, padding, mode="constant", constant_values=np.nan)

    # Replace diagonal with NAN
    np.fill_diagonal(A, np.nan)

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    # pylint: disable=unsubscriptable-object
    strides_w = (
        stride * A.strides[0],
        stride * A.strides[1],
        A.strides[0],
        A.strides[1],
    )
    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    return np.nanmean(A_w, axis=(2, 3))


def identify_optimal_scales(results, kernel_size=3, max_rand=1, basin_radius=1, include_boundary=True):
    """Identifies optimal scales in Severability.

    Robust scales are found in a sequential way. We first search for large diagonal blocks
    of low values in the 1-Rand(t, t') matrix that are located at local minima of its pooled
    diagonal, called block detection curve, and we obtain basins of fixed radius around
    these local minima. We then determine the minima of the Rand(t) curve for each basin,
    and these minima correspond to the robust partitions of the network.

    Args:
        sev_results (dict): the results from a Severability calculation
        kernel_size (int): size of kernel for average-pooling of the 1-Rand(t,t') matrix
        max_nvi (float): threshold for local minima of the pooled diagonal
        basin_radius (int): radius of basin around local minima of the pooled diagonal
        include_boundary (bool): if True, include the left and right boundaries of the
            block detection curve in the search for local minima

    Returns:
        result dictionary with two new keys: 'selected_partitions' and 'block_rand'

    """
    # get 1 - Rand(t) and 1 - Rand(t,t')
    rand_t = np.asarray(results["rand_t"])
    rand_tt = results["rand_ttprime"]

    # pool Rand(s,s')
    rand_tt_pooled = _pool2d_rand(
        rand_tt, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2)
    )

    block_rand = np.diag(rand_tt_pooled)[: len(rand_t)]
    results["block_rand"] = block_rand

    # replace nan at edges with 1, this allows to detect local minima at boundary
    block_rand_extended = block_rand.copy()
    block_rand_extended[~ (block_rand_extended >= 0)] = 1

    if include_boundary:
        # insert 1 to the left, this allows to detect local minima on left boundary
        block_rand_extended = np.insert(block_rand_extended,0,1)

        # append 1 to the right, this allows to detect local minima on right boundary
        block_rand_extended = np.append(block_rand_extended, 1)

    # find local minima on diagonal of pooled 1-Rand(s,s')
    basin_centers, _ = find_peaks(-block_rand_extended, height=-max_rand)
    # adjust indices after we insert 1 at the beginning
    basin_centers -= 1

    # add robust scales located in large 0 margins
    not_nan_ind = np.argwhere(~np.isnan(block_rand)).flatten()

    if (
        np.count_nonzero(
            np.around(block_rand[not_nan_ind[0] : not_nan_ind[0] + 2 * basin_radius + 1], 5)
        )
        == 0
    ):
        basin_centers = np.insert(basin_centers, 0, not_nan_ind[0] + basin_radius)

    if (
        np.count_nonzero(
            np.around(block_rand[not_nan_ind[-1] - 2 * basin_radius : not_nan_ind[-1] + 1], 5)
        )
        == 0
    ):
        basin_centers = np.append(basin_centers, not_nan_ind[-1] - basin_radius)
    # robust scales are minima of Rand(s) in basins
    robust_scales = set()
    for basin_center in basin_centers:
        # basins should not extend beyond domain of block detection curve
        basin = np.arange(
            max(basin_center - basin_radius, not_nan_ind[0]),
            min(basin_center + basin_radius + 1, not_nan_ind[-1]),
            dtype="int",
        )
        robust_scales.add(basin[np.argmin(rand_t[basin])])  
    
    # sort robust scales
    robust_scales = list(robust_scales)
    robust_scales.sort()

    # return with results dict
    results["selected_partitions"] = robust_scales

    return results