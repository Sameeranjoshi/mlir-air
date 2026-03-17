import re
from model_explorer import graph_builder

# Maps AIR graphNodeProperties colors to Model Explorer styles
AIR_COLOR_MAP = {
    'yellow':     graph_builder.GraphNodeStyle(backgroundColor='#FFD700', borderColor='#B8860B'),
    'crimson':    graph_builder.GraphNodeStyle(backgroundColor='#DC143C', borderColor='#8B0000'),
    'cyan':       graph_builder.GraphNodeStyle(backgroundColor='#00CED1', borderColor='#008B8B'),
    'chartreuse': graph_builder.GraphNodeStyle(backgroundColor='#7FFF00', borderColor='#228B22'),
}

AIR_COLOR_TO_TYPE = {
    'yellow':     'hierarchy',
    'crimson':    'control',
    'cyan':       'data (DMA)',
    'chartreuse': 'compute',
}


def get_node_style(color: str) -> graph_builder.GraphNodeStyle:
    return AIR_COLOR_MAP.get(color, graph_builder.GraphNodeStyle())


def color_to_op_type(color: str) -> str:
    return AIR_COLOR_TO_TYPE.get(color, 'unknown')


def parse_timing(label: str) -> tuple[str, tuple[int, int] | None]:
    """Extract timing [start-end] from node label, return (clean_label, (start, end))."""
    match = re.search(r'\[(\d+)-(\d+)\]', label)
    if match:
        clean = label[:match.start()].strip()
        return clean, (int(match.group(1)), int(match.group(2)))
    return label, None
