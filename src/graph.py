from graphviz import Digraph
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Any

def trace(root):
    """
    Traces the graph starting from the given root node and returns a set of all nodes and edges in the graph.

    Parameters:
    root (Node): The root node of the graph.

    Returns:
    tuple: A tuple containing two sets - the set of all nodes in the graph and the set of all edges in the graph.
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    """
    Creates a graph visualization of the computation graph rooted at the given node.

    Parameters:
    root (Node): The root node of the computation graph.

    Returns:
    dot (Digraph): The graph visualization in the form of a Digraph object.
    """
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'}) # LR = Left to Right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # create the root node
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        # create a fake op node
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    # todo
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render(filename='graph', directory='images', cleanup=True)
    return dot