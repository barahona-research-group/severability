# Severability
Code for the severability component quality function

Please cite:
Yu, Y.W., Delvenne, J.C., Yaliraki, S.N. and Barahona, M., 2020. Severability of mesoscale components and local time scales in dynamical networks. arXiv preprint arXiv:2006.02972.

## Installation

You can also install the source code of this package from GitHub directly by first cloning this repo with:
```bash
git clone https://github.com/barahona-research-group/severability 
``` 
To install the package, simply run (within the `severability` directory):
```bash
pip install . 
```

## Using the code

> :warning: **The code is based on deprecated `np.matrix`**: Be careful to use `np.matrix` instead of `np.ndarray` for adjacency and transition matrices!

To apply the Severability method to a graph defined by its adjacency matrix `A` (of type `np.matrix`!) we first compute its transition probability matrix:

```python
import severability

P = severability.transition_matrix(A)
```

We can then use Severability to compute a graph partition of overlapping clusters (also called "component cover"), which is a minimal cover of the graph with overlapping clusters described in Appendix D:

```python
partition = severability.minimal_component_cover(P,t=1)
```

The parameter `t` specifies the Markov time and in regular graphs, we expect that the partition gets coarser when increasing `t`. Note that `minimal_component_cover()` is not fully deterministic because it iterates through the nodes in a random order.

If we are only interested in the local community of a single node `i`, we can compute its unique "node component":

```python
cluster = severability.node_component(P,i=i,t=1)
```

We can also obtain a unique graph partition by combining the node components of all individual nodes, which we call the "node component cover":

```python
partition = severability.node_component_cover(P,t=1)
```

### Command line interface

Usage:
  ./severability.py \[-t MarkovTime\] \[-i InitialNode\] \[-s MAX_SIZE\] graph_file
  
If no initial node is chosen, then the program will find sufficiently many components to cover all nodes in graph_file. Otherwise, it will only find a single community containing the node seeded as InitialNode. The resolution factor of MarkovTime is by default set to 4.

## Examples

We provide examples in the `notebooks` directory.

To run an example from the command line, run the following:

* python examples/gen_ring_of_smallworlds.py --inner_num 25 > 2smallworlds.txt
* python severability.py 2smallworlds.txt

## Contributors

- Yun William Yu, GitHub: `yunwilliamyu <https://github.com/yunwilliamyu>`
- Dominik Schindler, GitHub: `d-schindler <https://github.com/d-schindler>`

We always look out for individuals that are interested in contributing to this open-source project. Even if you are just using `severability` and made some minor updates, we would be interested in your input. 

## Cite

Please cite our paper if you use this code in your own work:

```
@article{yuSeverabilityMesoscaleComponents2020,
  title = {Severability of Mesoscale Components and Local Time Scales in Dynamical Networks},
  author = {Yu, Yun William and Delvenne, Jean-Charles and Yaliraki, Sophia N. and Barahona, Mauricio},
  publisher = {arXiv},
  year = {2020},
  doi = {10.48550/arXiv.2006.02972},
  url = {http://arxiv.org/abs/2006.02972}
}
```

## Licence

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.