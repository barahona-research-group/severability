"""Utils for severability."""

import numpy as np


def orphan_nodes(partition, n_nodes):
    """
    Given a partition finds the orphan nodes
    """
    all_nodes = set(range(n_nodes))
    clustered_nodes = set()

    for cluster, _ in partition:
        for node in cluster:
            clustered_nodes.add(node)
    unclustered_nodes = all_nodes - clustered_nodes

    return list(unclustered_nodes)


def partition_to_matrix(partition, n_nodes, individuals=True):
    """
    Input:
        Partition
        Individuals: True or False - requirement if output is required to count orphan nodes as their own cluster or not

    Output:
        c x n matrix where inputs are severability of each node j in cluster i

    """

    U = np.zeros((len(partition), n_nodes))
    for i, (cluster, strength) in enumerate(partition):
        for node in cluster:
            U[i, node] = strength

    if individuals is True:
        unassigned_nodes = np.where(U.sum(axis=0) == 0)[0]
        new_rows = np.zeros((len(unassigned_nodes), U.shape[1]))
        for i, node in enumerate(unassigned_nodes):
            new_rows[i, node] = 1
        updated_U = np.vstack([U, new_rows])

        return updated_U
    else:
        return U
