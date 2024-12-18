from .options import Options


class FoamOptions(Options):
    """
    Options for foam generation.
    """

    radius_scale: float = 0.2  # foam particle radius w.r.t fluid particle radius
    color: tuple = (0.7, 0.7, 0.7, 0.7)
    spray_decay: float = 2.0  # The dissipation rate of spray foam
    foam_decay: float = 1.0  # The dissipation rate of foam foam
    bubble_decay: float = 5.0  # The dissipation rate of bubble foam
    k_foam: float = 1000.0  # amount of foam generated per frame


class CoacdOptions(Options):
    """
    Options for configuring coacd convex decomposition.
    Reference: https://github.com/SarahWeiii/CoACD
    """

    threshold: float = 0.1
    max_convex_hull: int = -1
    preprocess_mode: str = "auto"  # ['on', 'off', 'auto']
    preprocess_resolution: int = 30
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    pca: int = False
    merge: bool = True
    decimate: bool = False
    max_ch_vertex: int = 256
    extrude: bool = False
    extrude_margin: float = 0.01
    apx_mode: str = "ch"  # ['ch', 'box']
    seed: int = 0

    def __init__(self, **data):
        super().__init__(**data)
