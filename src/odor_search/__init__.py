"""odor_search — analysis pipeline for odor-driven Drosophila local-search assays.

Shared algorithms (arena/cylinder detection, trajectory assignment and cleaning,
kinematics, behavior segmentation) live in this package; each pipeline stage is
exposed as an ``odor-search`` CLI subcommand (see :mod:`odor_search.cli`).

Submodules deliberately are not imported here so ``import odor_search`` stays
lightweight and free of heavy optional dependencies (OpenCV, sleap-io, …).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
