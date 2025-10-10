from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

THIS_DIR = Path(__file__).resolve().parent

RESULTS_DIR = THIS_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
SYNERGY_DIR = IMAGES_DIR / "synergies"
RECON_DIR = IMAGES_DIR / "reconstruction"

# build folders if they do not already exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SYNERGY_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR.mkdir(parents=True, exist_ok=True)

def save_reconstruction_plot(fig: Figure, g: int) -> None:
    """Saves the joint angular velocity reconstruction error at grasp g to a .png file.

    Params:
        fig (Figure): matplotlib graph of the reconstruction error at grasp g.
        g (int): The grasping task shown in the figure.
    Returns:
        None.
    """
    fig.savefig(RECON_DIR / f"grasp{g + 1}.png")

def save_synergy_plot(fig: Figure, j: int) -> None:
    """
    """
    fig.savefig(SYNERGY_DIR / f"synergy{j + 1}.png")

def save_active_repeats(active_synergies: npt.NDArray, tol: float = 1e-6) -> None:
    """
    """
    # save in .npz
    np.savez(RESULTS_DIR / f"active_synergies_tol={tol:.0e}.npz", active_synergies=active_synergies)