"""Code for multiscale Severability."""

import multiprocessing
import numpy as np

from functools import partial
from tqdm import tqdm

import severability
from severability.utils import partition_to_matrix


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
        node_sev = [0 for i in range(N)]

        for i in range(N):
            total = 0
            count = 0
            for j in range(K):
                if part[j, i] > 0:
                    total += part[j, i]
                    count += 1
                else:
                    pass
            node_sev[i] = total / count

        return np.mean(node_sev)
    else:
        # compute unweighted mean severability over all clusters
        return np.mean([strength for _, strength in partition])


def multiscale_severability(
    P,
    t_max=2,
    n_tries=20,
    n_rand=10,
    n_workers=1,
    max_size=50,
    seed=42,
    filename="sev_results.pkl",
):
    """ "Compute multiscale severability."""

    scales = np.arange(1, t_max + 1)
    n_scales = len(scales)
    n_nodes = P.shape[0]

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
            all_partitions_t = pool.map(
                worker, streams[i * n_tries : (i + 1) * n_tries], chunksize=chunksize
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

        # compute average 1-Rand(t) for n_rand samples TODO: this already adds the orphans
        rand_t[i] = np.mean(
            severability.compute_rand_ttprime(
                all_partitions_t[:n_rand], n_nodes=n_nodes
            )
        )

        # count number and size of communities, including orphans
        n_communities[i] = len(optimal_partition_t)
        s_communities[i] = np.mean([len(cluster) for cluster, _ in optimal_partition_t])

    print("Compute 1-Rand(t,t') ...")
    rand_ttprime = severability.compute_rand_ttprime(partitions, n_nodes=n_nodes)

    # store results as dictionary
    results = {
        "scales": scales,
        "mean_size": s_communities,
        "n_communities": n_communities,
        "mean_sev": mean_severabilities,
        "rand_t": rand_t,
        "rand_ttprime": rand_ttprime,
        "partitions": partitions,
        "all_partitions": all_partitions,
    }

    severability.save_results(results, filename)

    return results
