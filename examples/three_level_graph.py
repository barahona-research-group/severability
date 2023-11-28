"""Generates benchmark graph adjacency matrix with 3 hierarchical levels"""
import numpy as np

def three_level_graph(k1=16, k2=4, k3=4, p1=0.8452, p2=0.0549, p3=0.0036):
    """Generates benchmark graph adjacency matrix with 3 hierarchical levels.
    Total number of nodes will be k1*k2*k3

    k1 (int):           Nodes in bottom level structure
    k2 (int):           k1 components in each k2 structure
    k3 (int):           k2 components in the k3 structure (which is the whole graph)
    p1 (float):         probability that two nodes in the same k1 component are
                        connected by an undirected edge
    p2 (float):         probability that two nodes in the same k2 component but
                        not the same k1 component are connected by an undirected
                        edge
    p3 (float):         probability that two nodes not in either the same k1 or
                        or k2 component are connected by an undirected edge

    Current parameters are set for about an average degree of 16.

    returns an adjacency matrix
    """
    size = k1 * k2 * k3
    k1_marker = []
    for i in range(k2 * k3):
        k1_marker.extend([i for _ in range(k1)])
    k2_marker = []
    for i in range(k3):
        k2_marker.extend([i for _ in range(k1 * k2)])
    A = np.matrix(np.zeros((size, size)))
    for i in range(size):
        for j in range(i):
            prob = p3
            if k2_marker[i] == k2_marker[j]:
                prob = p2
            if k1_marker[i] == k1_marker[j]:
                prob = p1
            if np.random.uniform() <= prob:
                A[i, j] = 1
    return A + A.transpose()
    