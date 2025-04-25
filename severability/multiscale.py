"""Code for multiscale Severability."""

import multiprocessing
import numpy as np

from functools import partial
from tqdm import tqdm

import severability
from severability.utils import partition_to_matrix
from severability.optimal_scales import identify_optimal_scales


def _get_chunksize(n_comp, pool):
    """Split jobs accross workers for speedup."""
    return max(1, int(n_comp / pool._processes))  # pylint: disable=protected-access


def _optimise(seed, P, t, max_size):
    return severability.component_cover(P, t, max_size, seed)


def _compute_mean_severability(partition, n_nodes, weighted=True):
    """ "Compute mean severability of partition."""
    if weighted is True:

        part = partition_to_matrix(partition, n_nodes, individuals=True)
        K, N = np.shape(part)

        mask = part > 0
        column_counts = mask.sum(axis=0)
        column_sums = part.sum(axis=0)

        node_sev = np.divide(column_sums, column_counts)

        return np.mean(node_sev)
    else:
        # compute unweighted mean severability over all clusters
        return np.mean([strength for _, strength in partition])


def multiscale_severability(
    A,
    t_min=2,
    t_max=10,
    even_times_only=True,
    n_tries=20,
    n_workers=1,
    max_size=50,
    with_optimal_scales=True,
    optimal_scales_kwargs=None,
    seed=42,
    filename="sev_results.pkl",
    verbose=False,
):
    """
    Compute multiscale severability for a given adjacency matrix.

    This function evaluates the severability of a graph across multiple scales
    by optimizing community partitions and computing severability metrics.
    It supports parallel processing for efficiency and can optionally identify
    optimal scales for severability.

    Args:
        A (numpy.ndarray): Adjacency matrix of the graph.
        t_min (int, optional): Minimum scale to evaluate. Defaults to 1.
        t_max (int, optional): Maximum scale to evaluate. Defaults to 2.
        even_times_only (bool, optional): If True, only even scales are evaluated. Defaults to True.
        n_tries (int, optional): Number of optimization attempts per scale. Defaults to 20.
        n_workers (int, optional): Number of parallel workers for multiprocessing. Defaults to 1.
        max_size (int, optional): Maximum size of communities during optimization. Defaults to 50. If -1, max_size
            is set to the number of nodes.
        with_optimal_scales (bool, optional): Whether to identify optimal scales. Defaults to True.
        optimal_scales_kwargs (dict, optional): Additional arguments for identifying optimal scales. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        filename (str, optional): File name to save the results. Defaults to "sev_results.pkl".
        verbose (bool, optional): If True, prints progress messages for each scale. Defaults to False.


    Returns:
        dict: A dictionary containing the following keys:
            - "run_params": Parameters used for the run.
            - "scales": Array of scales evaluated.
            - "mean_size": Mean size of communities at each scale.
            - "n_communities": Number of communities at each scale.
            - "mean_sev": Mean severability at each scale.
            - "rand_t": Average 1-Rand index for partitions at each scale.
            - "rand_ttprime": 1-Rand index between partitions across scales.
            - "partitions": Optimal partitions at each scale.
            - "all_partitions": All partitions generated during optimization.
            - Additional keys if `with_optimal_scales` is True.

    Notes:
        - The function uses multiprocessing to speed up the optimization process.
        - Severability metrics are computed using the `severability` module.
        - Results are saved to a file in pickle format for later use.
    """
    # check if A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be square.")
    # check if A is symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Adjacency matrix A must be symmetric.")

    # compute number of nodes
    n_nodes = A.shape[0]

    # define step size and increase t_min to even number if even_times_only
    if even_times_only:
        if t_min % 2 != 0:
            t_min += 1
        step = 2
    else:
        step = 1

    # define scales
    scales = np.arange(t_min, t_max + 1, step)
    n_scales = len(scales)

    # check max_size
    if max_size == -1:
        max_size = n_nodes

    # transform A to np.matrix and compute transition matrix
    P = severability.transition_matrix(np.matrix(A))

    # initialise random number generator
    parent_rng = np.random.default_rng(seed)
    # spawn independent random number generators from parent
    streams = parent_rng.spawn(n_scales * n_tries)

    # initialise results
    n_communities = np.zeros(n_scales)
    s_communities = np.zeros(n_scales)
    mean_severabilities = np.zeros(n_scales)
    rand_t = np.zeros(n_scales)
    rand_ttprime = np.zeros((n_scales, n_scales))
    partitions = []
    all_partitions = []

    print("Optimise severability ...")
    with multiprocessing.Pool(n_workers) as pool:
        # iterate through all scales
        for i in tqdm(range(len(scales))):
            t = scales[i]
            # define worker for parallel processing
            worker = partial(
                _optimise,
                P=P,
                t=t,
                max_size=max_size,
            )
            # repeate optimisation for n_tries in parallel
            chunksize = _get_chunksize(n_tries, pool)
            all_partitions_t = list(
                tqdm(
                    pool.imap(
                        worker,
                        streams[i * n_tries : (i + 1) * n_tries],
                        chunksize=chunksize,
                    ),
                    total=n_tries,
                    desc=f"Scale {scales[i]}",
                    disable=not verbose,
                )
            )
            all_partitions.append(all_partitions_t)

    print("Compute 1-Rand(t) ...")
    # iterate through all scales
    for i in tqdm(range(len(scales))):
        all_partitions_t = all_partitions[i]
        # compute mean severability for partitions obtained through different optimization tries
        all_mean_severabilities_t = np.zeros(n_tries)
        for j in range(n_tries):
            partition_t = all_partitions_t[j]
            all_mean_severabilities_t[j] = _compute_mean_severability(
                partition_t, n_nodes
            )
        # find optimal partition for t as the one with highest mean severability
        optimal_ind = np.argmax(all_mean_severabilities_t)
        mean_severabilities[i] = all_mean_severabilities_t[optimal_ind]
        optimal_partition_t = all_partitions_t[optimal_ind]

        # compute statistics for optimal partition
        partitions.append(optimal_partition_t)

        # compute average 1-Rand(t) TODO: this already adds the orphans
        rand_t[i] = np.mean(
            severability.compute_rand_ttprime(all_partitions_t, n_nodes=n_nodes)
        )

        # count number and size of communities, including orphans
        n_communities[i] = len(optimal_partition_t)
        s_communities[i] = np.mean([len(cluster) for cluster, _ in optimal_partition_t])

    print("Compute 1-Rand(t,t') ...")
    rand_ttprime = severability.compute_rand_ttprime(partitions, n_nodes=n_nodes)

    # store results as dictionary
    results = {
        "run_params": {
            "t_min": t_min,
            "t_max": t_max,
            "even_times_only": even_times_only,
            "n_tries": n_tries,
            "n_workers": n_workers,
            "max_size": max_size,
            "seed": seed,
        },
        "scales": scales,
        "mean_size": s_communities,
        "n_communities": n_communities,
        "mean_sev": mean_severabilities,
        "rand_t": rand_t,
        "rand_ttprime": rand_ttprime,
        "partitions": partitions,
        "all_partitions": all_partitions,
    }

    if with_optimal_scales:
        if optimal_scales_kwargs is None:
            optimal_scales_kwargs = {
                "kernel_size": max(3, int(0.1 * t_max)),
                "basin_radius": max(1, int(0.01 * t_max)),
            }
        results = identify_optimal_scales(results, **optimal_scales_kwargs)

    severability.save_results(results, filename)

    return results
