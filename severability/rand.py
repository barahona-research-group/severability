"""Code for Rand index to compute distance of soft partitions."""

import numpy as np

from severability.utils import orphan_nodes, partition_to_matrix

def mass_function(U):
    """Computes mass function for partition."""
    K, N = np.shape(U)
    # normalise matrices
    total = np.sum(U, axis=0)
    Unew = U / total

    K_U, N = np.shape(Unew)

    # calculate mass functions
    M_U = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            total_same = 0
            for k in range(K_U):
                total_same += Unew[k, i] * Unew[k, j]
            M_U[i, j] = total_same

    return M_U

def rand_similarity(M_U, M_V):
    """Computes Rand similarity between two (soft) partitions."""
    N = len(M_U)
    total = 0
    for i in range(N):
        for j in range(i, N):
            # add distance
            total += abs(M_U[i, j] - M_V[i, j])
    rho = 1 - total / (N * (N - 1) / 2)

    return rho

def compute_rand_ttprime(partitions, n_nodes):
    """Computes 1-Rand(t,t') for a sequence of soft partitions."""

    # deal with orphan
    for partition in partitions:
        # get orphans
        orphans = orphan_nodes(partition, n_nodes)
        for orphan in orphans:
            orpahn_cluster = [[orphan], 0.5]
            partition.append(orpahn_cluster)

    # compute mass functions
    partitions_mass_function = []
    for partition in partitions:
        partition_matrix = partition_to_matrix(partition, n_nodes, individuals=True)
        partitions_mass_function.append(mass_function(partition_matrix))

    # compute 1-Rand index
    rho = np.zeros((len(partitions), len(partitions)))
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            rho[i, j] = 1 - rand_similarity(
                partitions_mass_function[i], partitions_mass_function[j]
            )
            rho[j, i] = rho[i, j]

    return rho
