#!/usr/bin/env python
"""Code for optimizing the severability component quality function"""

from __future__ import print_function

__version__ = "0.0.1"

import sys
import argparse
import random
import numpy as np
import numpy.linalg as la

np.seterr(all="raise")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_3column_graph(fname, directed=False):
    """Loads a graph in 3-column tabular format, delimited by whitespace:

    1   2   0.5
    3   2   0.4
    1   3   0.2
    A2  2   1.5

    Where each line is an edge. If directed=True, will treat each edge as a
    directed arrow, otherwise just an undirected link.

    First and second columns are node names, and third column is weight. If
    the weight column is missing, will furnish a default weight of 1. (this also
    allows easy handling of unweighted graphs that lack edge weights)

    Skips any row with less than two columns, more than three columns, or where
    the third column cannot be parsed as a number.

    Self-loops are not double counted when undirected.

    Returns an adjacency matrix of a graph and a mapping from matrix indices to
    node names (names can be any strings, not just integers)."""
    with open(fname) as f:
        content = f.readlines()
    nodes = set()
    edges = []
    for l in content:
        row = l.split()
        # Get all nodes first
        if len(row) == 2:
            nodes.add(row[0])
            nodes.add(row[1])
            edges.append((row[0], row[1], 1))
        if len(row) == 3:
            if is_number(row[2]):
                nodes.add(row[0])
                nodes.add(row[1])
                edges.append((row[0], row[1], float(row[2])))

    name2ind = {}  # maps node names to numbers
    ind2name = {}  # maps node numbers to names
    for i, n in enumerate(nodes):
        name2ind[n] = i
        ind2name[i] = n

    # Generate graph
    size = len(nodes)
    A = np.matrix(np.zeros((size, size)))
    for e in edges:
        i = name2ind[e[0]]
        j = name2ind[e[1]]
        A[i, j] = A[i, j] + e[2]
        if not directed and (i != j):
            A[j, i] = A[j, i] + e[2]
    return (A, ind2name, name2ind)


def transition_matrix(adj):
    """Computes a Markov transition matrix from an adjacency matrix

    adj (np.matrix): n x n matrix specifying the connection pattern of a graph

    returns a transition matrix: an n x n matrix P where P = diag(A)^-1 * A"""
    diag = np.sum(adj, 1)
    trans_mat = adj / diag
    return trans_mat


def retention(P_C_power):
    """Given P_C^t, compute retention, which is just the sum of all entries
    divided by the number of rows, measuring the chance of a random walker
    to stay in the component given by C"""
    return np.sum(P_C_power) / P_C_power.shape[0]


def mixing(P_C_power):
    """Given P_C^t, computes the mixing, which is how different each row of
    P_C^t is from the average of the rows, after normalization of probabilities
    to 1, an approximation of the quasistationary distribution on the nodes of
    a random walker in C
    """
    diag = np.sum(P_C_power, 1)
    norm_mat = P_C_power / diag

    quasi_dist = np.average(norm_mat, 0)

    mixing = 1 - (
        (0.5 / P_C_power.shape[0]) * np.sum(np.absolute(norm_mat - quasi_dist))
    )
    return mixing


def severability_of_matrix_power(P_C_power):
    """Computes severability, which is 1/2 * (mixing + retention)"""
    diag = np.sum(P_C_power, 1)
    if 0 in diag:
        sev = 0  # Don't let disconnected matrices
    else:
        sev = (mixing(P_C_power) + retention(P_C_power)) / 2
    return sev


def severability_of_component(P, C, t):
    """Computes severability of component C with transition matrix P at Markov
    time t"""
    P_C = np.asarray(P[[[i] for i in C], C])  # Get submatrix
    if len(P_C) == 0:
        return 0
    else:
        P_C_power = la.matrix_power(P_C, t)  # P_C**t
        return severability_of_matrix_power(P_C_power)


def component_cover(P, t, max_size=50):
    """This is almost like partitioning a graph. We want to cover the
    entire network with components such that every node is either in a
    component, or is an "orphan"---i.e. it gets kicked out of every
    component we try to put it in by the KL steps. Orphans will not be
    returned in the output, but can be identified as the missing nodes
    not covered by any component.

    P (np.matrix):      Markov transition matrix
    t (int):            Markov time
    max_size (int):     stop the search at max_size

    returns list(component, severability)
    """
    remaining_nodes = set(range(P.shape[0]))
    ans = []
    potential_orphans = set()
    while len(remaining_nodes) > 0:
        n = random.sample(remaining_nodes, 1)[0]
        C, sev = component_optimise(P, [n], t, max_size)
        if len(remaining_nodes.intersection(C)) > 0:
            ans.append((C, sev))
            remaining_nodes.difference_update(C)
        if n in remaining_nodes:
            potential_orphans.add(n)
            remaining_nodes.remove(n)
    # For all potential orphans, try again using node_component, which
    # tries to keep the starting node in the community.
    orphans = set()
    while len(potential_orphans) > 0:
        n = random.sample(potential_orphans, 1)[0]
        C, sev = node_component(P, n, t, max_size)
        if len(potential_orphans.intersection(C)) > 0:
            ans.append((C, sev))
            potential_orphans.difference_update(C)
        if n in potential_orphans:
            orphans.add(n)
            potential_orphans.remove(n)
    return ans


def node_component(P, i, t, max_size=50):
    """Optimizes for the best component including a node:

    P (np.matrix):      Markov transition matrix
    i (int):            node to start from
    t (int):            Markov time
    max_size (int):     stop the search at max_size

    returns (component, severability)
    """
    linked_to = np.asarray(P[i, :]).nonzero()[0].tolist()
    neighbors = [item for item in linked_to if item not in [i]]
    # Order by highest severability directions to add nodes
    n_sorted = sorted(
        neighbors, key=lambda n: -1 * severability_of_component(P, [i, n], t)
    )

    for n in n_sorted:
        component, sev = component_optimise(P, [i, n], t, max_size)
        if i in component:
            return (component, sev)

    return ([], 0)


def connected_component(P, C):
    """Finds the max connected component for component C
    Currently uses BFS. Possibly want to reimplement with DFS?"""
    component = C
    linked_to = np.asarray(np.sum(P[component, :], 0)).nonzero()[0].tolist()
    new_component = list(set(C + linked_to))
    if len(new_component) == len(component):
        return new_component
    else:
        return connected_component(P, new_component)


def component_optimise(P, C, t, max_size=50):
    """Optimises for the best component starting from a community:

    P (np.matrix):      Markov transition matrix
    C list(int):        component to start from
    t (int):            Markov time
    max_size (int):     stop the search at max_size

    returns (component, severability)
    """
    # sys.stderr.write(".") This is to print the dots in output
    i = 1
    sev_max = 0
    C_max = C
    max_size = min(max_size, len(connected_component(P, C)))
    # Get up to max_size for the community using a 2-to-1 mix of greedy adds
    # and Kernighan-Lin steps
    while (len(C) < max_size) and (i < 3 * max_size):
        if (i % 3) == 0:
            (C, sev) = kernighan_lin_step(P, C, t)
        else:
            (C, sev) = greedy_add_step(P, C, t)
        if sev > sev_max:
            sev_max = sev
            C_max = C
        i = i + 1
    # Find local maximum for severability using optional Kernighan-Lin steps
    sev_improve = True
    C = C_max
    sev = sev_max
    while sev_improve:
        if len(C) == max_size:
            (C, sev) = greedy_remove_step(P, C, t)
        else:
            (C, sev) = kernighan_lin_step(P, C, t)
        if sev > sev_max:
            sev_max = sev
            C_max = C
        else:
            sev_improve = False
    return (C_max, sev_max)


def greedy_add_step(P, C, t):
    """Greedily adds a node to C such that the new severability is as high
    as possible. Note that a node *will* be added, even if any added node
    decreases the severability, unless there are no neighbors"""
    linked_to = np.asarray(np.sum(P[C, :], 0)).nonzero()[0].tolist()
    neighbors = [item for item in linked_to if item not in C]
    if len(neighbors) > 0:
        new_node = max(
            neighbors, key=lambda n: severability_of_component(P, C + [n], t)
        )
        C = C + [new_node]
    return (C, severability_of_component(P, C, t))


def greedy_remove_step(P, C, t):
    """Greedily removes a node to C such that the new severability is as high
    as possible. Note that a node *will* be removed, even if any removal
    decreases the severability"""
    removed_node = max(
        C, key=lambda n: severability_of_component(P, [x for x in C if x != n], t)
    )
    C = [x for x in C if x != removed_node]
    return (C, severability_of_component(P, C, t))


def kernighan_lin_step(P, C, t):
    """if can add a node, will either remove or add a node, optimising sev
    if can't add a node because no neighbors, will either remove a node or stay constant."""
    C_add, sev_add = greedy_add_step(P, C, t)
    C_rem, sev_rem = greedy_remove_step(P, C, t)
    if sev_add > sev_rem:
        return (C_add, sev_add)
    else:
        return (C_rem, sev_rem)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument("graph", help="input graph file")
    parser.add_argument("-i", "--initial", help="Starting node to find component")
    parser.add_argument(
        "-t", "--time", help="Markov time to run severability at", type=int, default=4
    )
    parser.add_argument(
        "-s",
        "--max-size",
        help="Maximum search size for the communities found",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    print(args)

    adj, ind2name, name2ind = load_3column_graph(args.graph)
    P = transition_matrix(adj)

    if args.initial is not None:
        print(name2ind[args.initial])
        C, sev = node_component(P, name2ind[args.initial], args.time, args.max_size)

        print(sev, len(C), [ind2name[n] for n in C])
    else:
        ans = component_cover(P, args.time, args.max_size)
        appearing = set()
        print("")
        for C, sev in ans:
            print(sev, "\t", len(C), "\t", sorted([ind2name[n] for n in C]))
            appearing.update(C)
        print("Nodes appearing:\t", len(appearing))
        print("Num communities:\t", len(ans))
    print("Markov time:\t", args.time)


if __name__ == "__main__":
    main()
