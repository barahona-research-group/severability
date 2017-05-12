#!/usr/bin/env python
'''Generates a ring of small worlds adjacency list'''
from __future__ import print_function

import sys
import argparse
import itertools
import numpy as np
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import gen_ring_of_rings as gror


def smallworld(nodes, degree, beta):
    '''Generates a small world network using the Watts-Strogatz method

    First generates an unweighted ring with weight_approx=degree

    Then for every edge, with probability beta randomly rewires it.
    '''
    A = set(gror.unweighted_ring(nodes, degree))
    A_copy = A
    for edge in A:
        if np.random.random() > beta:
            #B.append(edge)
            pass
        else:
            i, j, w = edge
            A_copy.remove(edge)
            # switch to numerical order
            if j < i:
                temp = j
                j = i
                i = temp
            while True:
                candidate = np.random.choice(nodes)
                if ((i, candidate, w) not in A_copy) and ((candidate, i, w) not in A_copy) and (candidate != i):
                    A_copy.add((i, candidate, w))
                    break
    return list(A_copy)

def ring_of_smallworlds(inner_num, outer_num, inner_degree=2, outer_links=1, beta=0.5):
    '''Generates a ring of small worlds with inner_num nodes in the inner worlds
    and outer_num rings. Small worlds are randomly connected in a ring with 
    outer_links links between adjacent small worlds.

    inner_num is a list of numbers
    outer_num is an int

    Returns an edgelist with weights specified by inner_degree and outer_weight
    '''
    edges = []
    rings = []
    curr_node = 1
    inner_it = itertools.cycle(inner_num)
    # Generate ring assignments
    for _ in range(outer_num):
        curr_inner_num = next(inner_it)
        rings.append(range(curr_node, curr_node + curr_inner_num))
        curr_node = curr_node + curr_inner_num

    # Generate within ring edges
    for ra in rings:
        edges.extend(smallworld(ra, inner_degree, beta))

    # Generate between ring edges
    if len(rings) == 2:
        for source, dest in [(0, 1)]:
            for _ in range(outer_links):
                edges.append((np.random.choice(rings[source]), np.random.choice(rings[dest]), 1))
    if len(rings) > 2:
        for source, dest in zip(range(len(rings)), range(1,len(rings)) + [0]):
            for _ in range(outer_links):
                edges.append((np.random.choice(rings[source]), np.random.choice(rings[dest]), 1))
    return edges

def main():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inner_num", help="Number of nodes in inner rings", 
            type=int, nargs="+")
    parser.add_argument("--outer_num", help="Number of rings",
            type=int, default=2)
    parser.add_argument("--inner_degree", help="Number of neighbors on each side to connecte nodes to",
            type=int, default=2)
    parser.add_argument("--outer_links", help="Number of links between adjacent small-worlds in the ring",
            type=int, default=1)
    parser.add_argument("--beta", help="rewiring probability within small-worlds",
            type=float, default=0.5)
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.inner_num == None:
        args.inner_num = [2]
    edges = ring_of_smallworlds(args.inner_num, args.outer_num, args.inner_degree,
        args.outer_links, args.beta)
    for e in edges:
        print(e[0], e[1], e[2])
    

if __name__ == "__main__":
    main()
