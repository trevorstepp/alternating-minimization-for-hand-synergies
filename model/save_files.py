from pathlib import Path
from matplotlib.figure import Figure

THIS_DIR = Path(__file__).resolve().parent

IMAGES_DIR = THIS_DIR / "images"
SYNERGY_DIR = IMAGES_DIR / "synergies"
RECON_DIR = IMAGES_DIR / "reconstruction"

# build folders if they do not already exist
#IMAGES_DIR.mkdir(parents=True, exist_ok=True)
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