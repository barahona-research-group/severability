#!/usr/bin/env python
'''Generates a ring of rings adjacency list'''
from __future__ import print_function

import sys
import argparse
import itertools
import numpy as np

def start_mid(ring):
    '''returns the 1st and middle value of the list'''
    start = ring[0]
    mid = ring[len(ring)//2]
    return (start, mid)

def unweighted_ring(nodes, weight_approx=2):
    '''Generates a single ring of nodes, but with where each node has degree
    2*weight_approx by connecting to weight_approx before neighbors and
    weight_approx after neighbors.

    For example, for weight_approx==2, each node i is connected to all of i+1,
    i+2, i-1, i-2. This simulates an ordinary ring where each link has weight 2
    using unweighted edges.
    '''
    if len(nodes) < (weight_approx*2 + 1):
        raise ValueError("not enough nodes. Weight_approx can't be more than number of potential neighbors")
    edges = []
    for i in range(weight_approx):
        edges.extend(zip(nodes, np.roll(nodes, i+1), itertools.repeat(1)))
    return edges

def weighted_ring(nodes, weight):
    '''Generates a single ring of nodes, but where each link has weight weight'''
    edges = []
    if len(nodes)==2:
        edges.append((nodes[0], nodes[1], weight))
    elif len(nodes)>2:
        edges.extend(zip(nodes, np.roll(nodes, 1), itertools.repeat(weight)))
    return edges

def ring_of_rings(inner_num, outer_num, inner_weight=2, outer_weight=1, weighted=True):
    '''Generates a ring of rings with inner_num nodes in the inner ring
    and outer_num rings. Nodes are connected in a circle within a ring
    and rings are connected in a circle from the opposite midpoints of
    each ring.

    Returns an edgelist with weights specified by inner_weight and outer_weight
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
    if weighted:
        gen_ring = weighted_ring
    else:
        gen_ring = unweighted_ring
    for ra in rings:
        edges.extend(gen_ring(ra, inner_weight))

    # Generate between ring edges
    if len(rings) > 1:
        sm_list = [start_mid(r) for r in rings]
        starts = [x[0] for x in sm_list]
        mids = [x[1] for x in sm_list]
        mids_rotated = [mids[-1]] + mids[:-1]
        edges.extend(zip(starts, mids_rotated, itertools.repeat(outer_weight)))
    return edges

def main():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inner_num", help="Number of nodes in inner rings", 
            type=int, nargs="+")
    parser.add_argument("--outer_num", help="Number of rings",
            type=int, default=2)
    parser.add_argument("--inner_weight", help="Weight of within node links",
            type=int, default=2)
    parser.add_argument("--outer_weight", help="Weight of links connecting rings",
            type=int, default=1)
    parser.add_argument("--unweighted", help="Approximates inner_weight with more "
            "edges not just nearest neighbor but the next ones",
            action="store_true")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.inner_num == None:
        args.inner_num = [2]
    edges = ring_of_rings(args.inner_num, args.outer_num, args.inner_weight,
        args.outer_weight, weighted = not args.unweighted)
    for e in edges:
        print(e[0], e[1], e[2])
    

if __name__ == "__main__":
    main()
