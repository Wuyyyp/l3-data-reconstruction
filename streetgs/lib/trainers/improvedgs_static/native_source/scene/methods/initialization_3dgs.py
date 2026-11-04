"""
Gaussian model initialization for the 3DGS scene representation.

The training script calls this before Scene loads the point cloud. The helper
chooses the starting SH degree for the selected method and creates the
GaussianModel instance with the configured optimizer type.
"""
from typing import Any

from scene.gaussian_model import GaussianModel as GaussianModel3DGS


def resolve_initial_sh_degree_3dgs(dataset: Any, opt: Any) -> int:
    """
        Choose the starting SH degree for the Gaussian model.

        MiniGS starts from degree 0 and restores higher SH capacity later; other
        methods start from the dataset-configured degree.
    """
    if str(getattr(opt, "training_method", "3dgs")).lower() == "minigs":
        return 0
    return int(dataset.sh_degree)


def build_gaussian_model_3dgs(dataset: Any, opt: Any) -> GaussianModel3DGS:
    """
        Create a GaussianModel and attach method-specific initialization flags.
    """
    gaussian_model = GaussianModel3DGS(
        resolve_initial_sh_degree_3dgs(dataset, opt),
        opt.optimizer_type,
    )
    gaussian_model.initial_opacity = float(opt.initial_opacity)
    gaussian_model.use_mcmc_initialization = (
        str(getattr(opt, "training_method", "3dgs")).lower() == "mcmc"
    )
    return gaussian_model
