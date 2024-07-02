"""Code for multiscale Severability."""

import numpy as np

from tqdm import tqdm

from severability import node_component


def node_component_cover(P, t, max_size=50, disable_tqdm=True):
    """Compute node component cover."""

    if not isinstance(P, np.matrix):
        raise TypeError("Transition matrix expects np.matrix type.")

    cover_dict = {}

    # iterate through all nodes
    for i in tqdm(range(P.shape[0]), disable=disable_tqdm):
        # compute node component for i
        component, sev = node_component(P, i, t, max_size)
        # add nonempty components to cover as frozensets to avoid duplicates
        if len(component) > 0:
            cover_dict[frozenset(component)] = sev

    # return cover as list of tuples of component and its severability
    cover = []
    for component, sev in cover_dict.items():
        cover.append((list(component), sev))

    return cover
