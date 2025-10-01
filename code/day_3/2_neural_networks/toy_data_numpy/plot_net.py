import math
import os

import pygraphviz as pgv


# We use a weight_normalisation to make visualization clearer
# we want the edge to be :
# - thick when the amplitude of the weight is high
# - thin when the amplitude of the weight is small
# - no weight should be visible when the weight has an amplitude of 0
# So we use some symetrical function, whose value in 0 is 0
def edge_width(weight):
    return 10 * (1 / (1 + math.exp(-abs(weight))) - 1 / 2)


# We would also like to visualize the sign og the weights
# To do so, we will use the color of the edges
def edge_color(weight):
    if weight >= 0:
        # yellow
        # color taken from a colorpicker
        return "#f4c441"
    else:
        # blue
        return "#42a7f4"


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def show_net(
    step,
    weights_1,
    weights_2,
    input_dim,
    hidden_dim,
    output_dim,
    figname,
    dir_name,
    graph_name,
    loss,
    Learning_rate,
):
    """
    Function to print a screenshot of a neural network at a given
    optimization step
    In these examples we will use networks with only one hidden layer,
    however the method can be applied to bigger networks.
    """
    # Initialiaze a graph object from the pygraphviz lib.
    # We will indeed use a graph visualization tool to monitor
    # the neural network.
    # The rankdir keyword argument plot the graph from left to right
    Net = pgv.AGraph(rankdir="LR")

    # input nodes
    input_nodes = ["i" + str(x) for x in range(input_dim)]
    Net.add_nodes_from(input_nodes, color="black")

    # hidden layer
    hidden_nodes = ["h" + str(x) for x in range(hidden_dim)]
    Net.add_nodes_from(hidden_nodes, color="black")

    # output layer
    output_nodes = ["o" + str(x) for x in range(output_dim)]
    Net.add_nodes_from(output_nodes, color="black")

    # draw edges
    # beetween the input layer and the hidden layer
    for input_node in range(input_dim):
        for hidden_node in range(hidden_dim):
            # get the weight of this edge
            weight = weights_1[input_node, hidden_node]
            # we need to give the string name of the node
            # to graphviz
            input_node_name = "i" + str(input_node)
            hidden_node_name = "h" + str(hidden_node)
            Net.add_edge(
                input_node_name,
                hidden_node_name,
                color=edge_color(weight),
                penwidth=edge_width(weight),
            )

    # between the hidden layer and the output layer
    for hidden_node in range(hidden_dim):
        for output_node in range(output_dim):
            weight = weights_2[hidden_node, output_node]
            hidden_node_name = "h" + str(hidden_node)
            output_node_name = "o" + str(output_node)
            Net.add_edge(
                hidden_node_name,
                output_node_name,
                color=edge_color(weight),
                penwidth=edge_width(weight),
            )

    # Add general info on the network
    # Step
    step_label = "step ----- " + "{:05d}".format(step)
    Net.add_node("step_node", shape="box", label=step_label, color="#9542f4")

    # Loss
    loss_label = "loss ----- " + "{:012f}".format(round(loss, 4))
    Net.add_node("loss_node", shape="box", label=loss_label, color="#9542f4")

    # Learning rate
    rate_label = "rate ----- " + str(Learning_rate)
    Net.add_node("rate_node", shape="box", label=rate_label, color="#9542f4")

    # Here we artificially add edges between theses nodes
    # but it is just useful to force graphviz to space
    # the layers and make visualization clearer.
    Net.add_edge("step_node", "loss_node", penwidth=0.0)
    Net.add_edge("loss_node", "rate_node", penwidth=0.0)

    # save stuff
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    Net.layout("dot")
    Net.draw(dir_name + graph_name + ".pdf")
