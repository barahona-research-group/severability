"""Code for plotting."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as _plot
except ImportError:  # pragma: no cover
    print("Interactive pliotting requires plotly.")

try:
    import graph_tool.all as gt 
except ImportError:  # pragma: no cover
    print("Plotting of overlapping communities requires graph-tool.")
    

import networkx as nx

from severability.utils import partition_to_matrix

########################################
### Code for plotting optimal scales ###
########################################

def plot_scan(
    all_results,
    figsize=(6, 5),
    figure_name="scan_results.pdf",
    use_plotly=False,
    live=True,
    plotly_filename="scan_results.html",
):
    """Plot results of severability with matplotlib or plotly.

    Args:
        all_results (dict): results of severability scan
        figsize (tuple): matplotlib figure size
        figure_name (str): name of matplotlib figure
        use_plotly (bool): use matplotlib or plotly backend
        live (bool): for plotly backend, open browser with pot
        plotly_filename (str): filename of .html figure from plotly
    """
    if len(all_results["scales"]) == 1:  # pragma: no cover
        return None

    if use_plotly:
        return plot_scan_plotly(all_results, live=live, filename=plotly_filename)
    return plot_scan_plt(
        all_results, figsize=figsize, figure_name=figure_name
    )


def plot_scan_plotly(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    all_results,
    live=False,
    filename="clusters.html",
):
    """Plot results of pygenstability with plotly."""
    scales = all_results["scales"]

    hovertemplate = str("<b>scale</b>: %{x:.2f}, <br>%{text}<extra></extra>")

    if "rand_t" in all_results:
        rand_t_data = all_results["rand_t"]
        rand_t_opacity = 1.0
        rand_t_title = "1 - Rand(t)"
        rand_t_ticks = True
    else:  # pragma: no cover
        rand_t_data = np.zeros(len(scales))
        rand_t_opacity = 0.0
        rand_t_title = None
        rand_t_ticks = False

    text = [
        f"""Number of communities: {n}, <br> Severability: {np.round(s, 3)},
        <br> Rand(t): {np.round(r_t, 3)}, <br> Index: {i}"""
        for n, s, r_t, i in zip(
            all_results["n_communities"],
            all_results["mean_sev"],
            rand_t_data,
            scales,
        )
    ]
    ncom = go.Scatter(
        x=scales,
        y=all_results["n_communities"],
        mode="lines+markers",
        hovertemplate=hovertemplate,
        name="Number of communities",
        xaxis="x2",
        yaxis="y4",
        text=text,
        marker_color="red",
    )

    if "rand_ttprime" in all_results:
        z = all_results["rand_ttprime"]
        showscale = True
        tprime_title = "1-Rand(t,t')"
    else:  # pragma: no cover
        z = np.nan + np.zeros([len(scales), len(scales)])
        showscale = False
        tprime_title = None

    ttprime = go.Heatmap(
        z=z,
        x=scales,
        y=scales,
        colorscale="YlOrBr_r",
        yaxis="y2",
        xaxis="x2",
        hoverinfo="skip",
        colorbar={"title": "1-Rand(t,t')", "len": 0.2, "yanchor": "middle", "y": 0.5},
        showscale=showscale,
    )
    if "mean_sev" in all_results:
        sev = go.Scatter(
            x=scales,
            y=all_results["mean_sev"],
            mode="lines+markers",
            hovertemplate=hovertemplate,
            text=text,
            name="Severability",
            marker_color="blue",
        )

    r_t = go.Scatter(
        x=scales,
        y=rand_t_data,
        mode="lines+markers",
        hovertemplate=hovertemplate,
        text=text,
        name="1-Rand(t)",
        yaxis="y3",
        xaxis="x",
        marker_color="green",
        opacity=rand_t_opacity,
    )

    layout = go.Layout(
        yaxis={
            "title": "Severability",
            "title_font": {"color": "blue"},
            "tickfont": {"color": "blue"},
            "domain": [0.0, 0.28],
        },
        yaxis2={
            "title": tprime_title,
            "title_font": {"color": "black"},
            "tickfont": {"color": "black"},
            "domain": [0.32, 1],
            "side": "right",
            "range": [scales[0], scales[-1]],
        },
        yaxis3={
            "title": rand_t_title,
            "title_font": {"color": "green"},
            "tickfont": {"color": "green"},
            "showticklabels": rand_t_ticks,
            "overlaying": "y",
            "side": "right",
        },
        yaxis4={
            "title": "Number of communities",
            "title_font": {"color": "red"},
            "tickfont": {"color": "red"},
            "overlaying": "y2",
        },
        xaxis={"range": [scales[0], scales[-1]]},
        xaxis2={"range": [scales[0], scales[-1]]},
    )

    fig = go.Figure(data=[sev, ncom, r_t, ttprime], layout=layout)
    fig.update_layout(xaxis_title="Scale")

    if filename is not None:
        _plot(fig, filename=filename, auto_open=live)
    return fig, layout


def _plot_number_comm(all_results, ax, scales):
    """Plot number of communities."""
    ax.plot(scales, all_results["n_communities"], "-", c="C3", label="size", lw=2.0)
    ax.set_ylim(0, 1.1 * max(all_results["n_communities"]))
    ax.set_ylabel("# clusters", color="C3")
    ax.tick_params("y", colors="C3")



def _plot_ttprime(all_results, ax, scales):
    """Plot ttprime."""
    contourf_ = ax.contourf(scales, scales, all_results["rand_ttprime"], cmap="YlOrBr_r", extend="min")
    ax.set_ylabel(r"$t^\prime$")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.axis([scales[0], scales[-1], scales[0], scales[-1]])
    ax.set_xlabel(r"$t$")

    axins = inset_axes(
        ax,
        width="3%",
        height="40%",
        loc="lower left",
        bbox_to_anchor=(0.05, 0.45, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axins.tick_params(labelsize=7)
    plt.colorbar(contourf_, cax=axins, label="Rand(t,t')")


def _plot_rand_t(all_results, ax, scales):
    """Plot 1-Rand(t)"""
    ax.plot(scales, all_results["rand_t"], "-", lw=2.0, c="C2", label="1-Rand(t)")

    ax.yaxis.tick_right()
    ax.tick_params("y", colors="C2")
    ax.set_ylabel(r"1-Rand(t)", color="C2")
    ax.axhline(1, ls="--", lw=1.0, c="C2")
    ax.axis([scales[0], scales[-1], 0.0, np.max(all_results["rand_t"]) * 1.1])
    ax.set_xlabel(r"$t$")


def _plot_sev(all_results, ax, scales):
    """Plot severability."""
    ax.plot(scales, all_results["mean_sev"], "-", label=r"Severability", c="C0")
    ax.tick_params("y", colors="C0")
    ax.set_ylabel("Severability", color="C0")
    ax.set_ylim(0.9*min(all_results["mean_sev"]), 1.1 * max(all_results["mean_sev"]))
    ax.yaxis.set_label_position("left")


def _plot_optimal_scales(all_results, ax, scales, ax1, ax2):
    """Plot stability."""
    ax.plot(
        scales,
        all_results["block_rand"],
        "-",
        lw=2.0,
        c="C4",
        label="Block Rand",
    )
    ax.plot(
        scales[all_results["selected_partitions"]],
        all_results["block_rand"][all_results["selected_partitions"]],
        "o",
        lw=2.0,
        c="C4",
        label="optimal scales",
    )

    ax.tick_params("y", colors="C4")
    ax.set_ylabel("Block Rand", color="C4")
    ax.yaxis.set_label_position("left")
    ax.set_xticks(scales)
    ax.set_xlabel(r"$t$")

    for scale in scales[all_results["selected_partitions"]]:
        ax.axvline(scale, ls="--", color="C4")
        ax1.axvline(scale, ls="--", color="C4")
        ax2.axvline(scale, ls="--", color="C4")


def plot_scan_plt(all_results, figsize=(6, 5), figure_name="scan_results.svg"):
    """Plot results of severability with matplotlib."""
    scales = all_results["scales"]
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 1.0, 0.5])
    gs.update(hspace=0)
    axes = []

    if "rand_ttprime" in all_results:
        ax0 = plt.subplot(gs[1, 0])
        axes.append(ax0)
        _plot_ttprime(all_results, ax=ax0, scales=scales)
        ax1 = ax0.twinx()
    else:  # pragma: no cover
        ax1 = plt.subplot(gs[1, 0])

    axes.append(ax1)
    ax1.set_xticks([])

    _plot_rand_t(all_results, ax=ax1, scales=scales)

    if "rand_ttprime" in all_results:
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

    ax2 = plt.subplot(gs[0, 0])

    if "mean_sev" in all_results:
        _plot_sev(all_results, ax=ax2, scales=scales)
        ax2.set_xticks([])
        axes.append(ax2)

    if "rand_t" in all_results:
        ax3 = ax2.twinx()
        _plot_number_comm(all_results, ax=ax3, scales=scales)
        axes.append(ax3)

    if "block_rand" in all_results:
        ax4 = plt.subplot(gs[2, 0])
        _plot_optimal_scales(all_results, ax=ax4, scales=scales, ax1=ax1, ax2=ax2)
        axes.append(ax4)

    for ax in axes:
        ax.set_xlim(scales[0], scales[-1])


    if figure_name is not None:
        plt.savefig(figure_name)
    

##################################################
### Code for plotting pie chart visualisations ###
##################################################

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



def plot_pie_graph(partition, adj, seed = 6):
    """
    Input:
        U: partition matrix 
        adj: adjacency matrix
        seed controlling graph layout
        
    Output:
        plots the visualisation
    """
    n_nodes = adj.shape[0]
    U = partition_to_matrix(partition, n_nodes, individuals = True)
    graph, pos = graph_position(adj, seed)
    vprops = vertex_properties(graph, U)
    gt.graph_draw(graph, pos = pos, vprops = vprops)
    
    