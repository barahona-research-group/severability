# severability
Code for the severability component quality function

Please cite:
Yu, Y.W., Delvenne, J.C., Yaliraki, S.N. and Barahona, M., 2020. Severability of mesoscale components and local time scales in dynamical networks. arXiv preprint arXiv:2006.02972.

Requires:
* numpy

Usage:
  ./severability.py \[-t MarkovTime\] \[-i InitialNode\] \[-s MAX_SIZE\] graph_file
  
If no initial node is chosen, then the program will find sufficiently many components to cover all nodes in graph_file. Otherwise, it will only find a single community containing the node seeded as InitialNode. The resolution factor of MarkovTime is by default set to 4.

To run an example, run the following:

* python examples/gen_ring_of_smallworlds.py 25 > 2smallworlds.txt
* python severability.py 2smallworlds.txt
