import pydot
from model_explorer import graph_builder
from .air_styling import get_node_style, color_to_op_type, parse_timing

# pydot reserved names that are not real nodes
_PYDOT_RESERVED = {'node', 'edge', 'graph'}


def parse_air_dot(dot_path: str) -> list[graph_builder.Graph]:
    gv_graphs = pydot.graph_from_dot_file(dot_path)
    if not gv_graphs:
        return []

    graphs = []
    for i, gv_graph in enumerate(gv_graphs):
        graph_id = gv_graph.get_name().strip('"') or f'graph_{i}'
        graph = graph_builder.Graph(id=graph_id)

        node_map: dict[str, graph_builder.GraphNode] = {}
        _collect_nodes(gv_graph, node_map, namespace='')
        _collect_edges(gv_graph, node_map)

        graph.nodes.extend(node_map.values())
        graphs.append(graph)

    return graphs


def _get_attr(attrs: dict, key: str, default: str = '') -> str:
    val = attrs.get(key, default)
    if isinstance(val, str):
        return val.strip('"')
    return default


def _collect_nodes(gv_graph, node_map: dict, namespace: str):
    """Recursively collect nodes; subgraph clusters become namespace prefixes."""
    for gv_node in gv_graph.get_nodes():
        name = gv_node.get_name().strip('"')
        if name in _PYDOT_RESERVED:
            continue

        attrs = gv_node.obj_dict.get('attributes', {})
        raw_label = _get_attr(attrs, 'label', name).replace('\\n', '\n')
        color = _get_attr(attrs, 'color')

        op_name, timing = parse_timing(raw_label)
        # Use the first line of the label as the display label
        display_label = op_name.split('\n')[0].strip()

        gn = graph_builder.GraphNode(id=name, label=display_label, namespace=namespace)
        gn.style = get_node_style(color)

        if color:
            gn.attrs.append(graph_builder.KeyValue(key='op_type', value=color_to_op_type(color)))
        # Include any extra label lines (e.g. "(L1, 256, i32)") as an attribute
        extra = '\n'.join(op_name.split('\n')[1:]).strip()
        if extra:
            gn.attrs.append(graph_builder.KeyValue(key='details', value=extra))
        if timing:
            gn.attrs.append(graph_builder.KeyValue(key='start_time', value=str(timing[0])))
            gn.attrs.append(graph_builder.KeyValue(key='end_time', value=str(timing[1])))
            gn.attrs.append(graph_builder.KeyValue(key='duration', value=str(timing[1] - timing[0])))

        node_map[name] = gn

    # Recurse into subgraph clusters -> namespace hierarchy
    for sub in gv_graph.get_subgraphs():
        sub_attrs = sub.obj_dict.get('attributes', {})
        sub_label = _get_attr(sub_attrs, 'label', sub.get_name())
        child_ns = f'{namespace}/{sub_label}' if namespace else sub_label
        _collect_nodes(sub, node_map, child_ns)


def _collect_edges(gv_graph, node_map: dict):
    """Recursively collect edges from graph and all subgraphs."""
    for edge in gv_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        if src in node_map and dst in node_map:
            node_map[dst].incomingEdges.append(
                graph_builder.IncomingEdge(sourceNodeId=src)
            )
    for sub in gv_graph.get_subgraphs():
        _collect_edges(sub, node_map)
