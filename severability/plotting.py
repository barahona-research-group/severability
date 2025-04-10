""" code for plotting pie chart visualisations"""

import graph_tool.all as gt 
import networkx as nx
from severability.utils import partition_to_matrix
import numpy as np

def matrix_to_gt(adj):
    """
    Input: 
        Adjacency matrix 
  
    Output:
        Graph-tools graph
    """
    G_gt = gt.Graph(directed = False)
    G_nx = nx.from_numpy_array(adj)
    # Add nodes to the Graph-Tool graph
    for node in G_nx.nodes:
        G_gt.add_vertex()

    # Add edges to the Graph-Tool graph
    for edge in G_nx.edges:
        G_gt.add_edge(edge[0], edge[1])
    return (G_gt)



def graph_position(adj, seed):
    """
    Input: 
        Adjacency matrix
        
    Output:
        Graph layout
    """
    G_gt = matrix_to_gt(adj)
    
    # we turn it into a network x graph to set seed and then into a graph tools graph for plotting 
    G_nx = nx.Graph()
    for v in G_gt.vertices():
        G_nx.add_node(int(v))
        
    for e in G_gt.edges():
        G_nx.add_edge(int(e.source()), int(e.target()))
        
        
    pos_nx = nx.spring_layout(G_nx, seed=seed)
    pos_gt = G_gt.new_vertex_property("vector<double>")

    # Assign positions from networkx to graph_tool
    for v in G_gt.vertices():
        pos_gt[v] = pos_nx[int(v)]
    
    pos = gt.sfdp_layout(G_gt, pos=pos_gt)
    return G_gt, pos


def compute_pie_fraction(U):
    """
    Input: 
        Partition matrix 
    Output:
        pie_frac : A list of lists, where each inner list contains the fractions of the total severability for each community at a node.
        num_of_communities : number of communities that each node is a member of
    """
    K, N = np.shape(U)
    pie_frac = [0 for i in range (N)]
    num_of_communities = [0 for i in range (N)]
    
    for i in range (N):
        
        pie = [0 for j in range (K)]
        total = sum(U[:,i])
        if total == 0:
            pass
        else:
            for j in range (K):
                pie[j] = U[j,i]/total
        pie_frac[i] = pie
        num_of_communities[i] = np.count_nonzero(pie) 
    return pie_frac, num_of_communities



def vertex_properties(G, U):
    """
    Creates vertex properties for a graph based on pie chart fractions for each node.
    Input:
        G (graph_tool.Graph): The graph object.
        U: the partition matrix
    
    Output:
        dict: A dictionary containing the vertex properties for shape, pie fractions, size, color, and border.
    """
    
    pie_frac, number_of_communities = compute_pie_fraction(U)
    node_sizes = [15 for x in number_of_communities]  

    # Identify orphan nodes
    orphan_nodes = set()
    for k, row in enumerate(U):
        nonzero_indices = np.nonzero(row)[0]
        if len(nonzero_indices) == 1 and row[nonzero_indices[0]] == 0.5:
            orphan_nodes.add(nonzero_indices[0])

    # Create vertex properties
    pie_frac_property = G.new_vertex_property("vector<double>")
    shape_property = G.new_vertex_property("string")
    color_property = G.new_vertex_property("vector<float>") 
    border_width_property = G.new_vertex_property("double")
    size_property = G.new_vertex_property("double")

    for v in G.vertices():
        idx = int(v)
        pie_frac_property[v] = pie_frac[idx]

        if idx in orphan_nodes:
            shape_property[v] = "square"
            size_property[v] = node_sizes[idx] + 10  
            color_property[v] = [1.0, 1.0, 1.0, 1.0]  # white
            border_width_property[v] = 2.0
        else:
            shape_property[v] = "pie"
            size_property[v] = node_sizes[idx]
            color_property[v] = [0.0, 0.0, 0.0, 0.0]  # transparent
            border_width_property[v] = 0.0

    vprops = {
        "shape": shape_property,
        "pie_fractions": pie_frac_property,
        "size": size_property,
        "fill_color": color_property,
        "pen_width": border_width_property,
    }
    return vprops



def plot_pie_graph(partition, adj, n_nodes, seed = 6):
    """
    Input:
        U: partition matrix 
        adj: adjacency matrix
        seed controlling graph layout
        
    Output:
        plots the visualisation
    """
    U = partition_to_matrix(partition, n_nodes, individuals = True)
    graph, pos = graph_position(adj, seed)
    vprops = vertex_properties(graph, U)
    gt.graph_draw(graph, pos = pos, vprops = vprops)
    
    