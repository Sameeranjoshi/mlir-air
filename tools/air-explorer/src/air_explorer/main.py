from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs
from .dot_parser import parse_air_dot


class AIRExplorerAdapter(Adapter):
    metadata = AdapterMetadata(
        id='air_explorer',
        name='AIR Dependency Graph Explorer',
        description='Visualize AIR ACDG dependency graphs with hierarchy, timing, and op-type coloring',
        source_repo='https://github.com/Xilinx/mlir-air',
        fileExts=['dot'],
    )

    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        graphs = parse_air_dot(model_path)
        return {'graphs': graphs}
