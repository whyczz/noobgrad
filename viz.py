"""
Graphviz visualization for Scalar computation graphs
"""

from graphviz import Digraph
from ascalar import Scalar


def trace(root: Scalar) -> tuple[set[Scalar], set[tuple[Scalar, Scalar]]]:
    """
    Build a set of all nodes and edges in the computation graph.

    Returns:
        nodes: set of all Scalar objects in the graph
        edges: set of (parent, child) tuples
    """
    nodes, edges = set(), set()

    def build(v: Scalar):
        if v not in nodes:
            nodes.add(v)
            if v._parents is not None:
                for parent in v._parents:
                    if parent is not None:  # Handle unary ops like __neg__
                        edges.add((parent, v))
                        build(parent)

    build(root)
    return nodes, edges


def draw_dot(root: Scalar, format: str = "svg", rankdir: str = "TB") -> Digraph:
    """
    Draw a computation graph for a Scalar.

    Args:
        root: The final Scalar to visualize (typically the output/loss)
        format: Output format ('svg', 'png', 'pdf', etc.)
        rankdir: Graph direction - 'TB' (top-bottom), 'LR' (left-right)

    Returns:
        Digraph object that can be rendered or displayed

    Example:
        >>> a = Scalar(2)
        >>> b = Scalar(3)
        >>> c = a * b + a
        >>> dot = draw_dot(c)
        >>> dot.render('computation_graph', view=True)
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    nodes, edges = trace(root)

    # Add nodes for each Scalar
    for n in nodes:
        # Create a unique ID for this node based on its memory address
        uid = str(id(n))

        # Draw the scalar value as a record node
        label = f"{{ value: {n.value:.4f} }}"
        dot.node(name=uid, label=label, shape="record")

        # If this node was created by an operation, add the operation node
        if n._parent_op:
            op_uid = uid + n._parent_op
            dot.node(name=op_uid, label=n._parent_op, shape="circle")
            # Connect operation to this node
            dot.edge(op_uid, uid)

    # Add edges from parents to operations
    for parent, child in edges:
        if child._parent_op:
            op_uid = str(id(child)) + child._parent_op
            dot.edge(str(id(parent)), op_uid)

    return dot
