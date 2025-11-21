from pathlib import Path
from matplotlib.figure import Figure
import scipy.io as sio
import numpy as np
import numpy.typing as npt

THIS_DIR = Path(__file__).resolve().parent

RESULTS_DIR = THIS_DIR / "results"
NPZ_SYNERGIES_DIR = RESULTS_DIR / "npz_saved_synergies"
IMAGES_DIR = RESULTS_DIR / "images"
SYNERGY_DIR = IMAGES_DIR / "synergies"
RECON_DIR = IMAGES_DIR / "reconstruction"
ASL_DIR = IMAGES_DIR / "asl_reconstruction"

# build folders if they do not already exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NPZ_SYNERGIES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SYNERGY_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR.mkdir(parents=True, exist_ok=True)
ASL_DIR.mkdir(parents=True, exist_ok=True)

def get_subj_dir(base_dir: Path, subject: str) -> Path:
    """Creates a subject directory inside the base directory.
    """
    subj_dir = base_dir / subject
    subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir

def save_reconstruction_plot(fig: Figure, g: int, subject: str) -> None:
    """Saves the joint angular velocity reconstruction error at grasp g to a .png file.

    Params:
        fig (Figure): matplotlib graph of the reconstruction error at grasp g.
        g (int): The grasping task shown in the figure.
    Returns:
        None.
    """
    subj_dir = get_subj_dir(RECON_DIR, subject)
    fig.savefig(subj_dir / f"grasp{g + 1}.png")

def save_synergy_plot(fig: Figure, j: int, subject: str) -> None:
    """
    """
    subj_dir = get_subj_dir(SYNERGY_DIR, subject)
    fig.savefig(subj_dir / f"synergy{j + 1}.png")

def save_asl_plot(fig: Figure, g: int, subject: str) -> None:
    """
    """
    subj_dir = get_subj_dir(ASL_DIR, subject)
    fig.savefig(subj_dir / f"asl_grasp{g + 1}.png")

def save_active_repeats(active_synergies: npt.NDArray, subject: str, tol: float = 1e-6) -> None:
    """
    """
    subj_dir = get_subj_dir(NPZ_SYNERGIES_DIR, subject)
    # save in .npz
    np.savez(subj_dir / f"active_synergies_tol={tol:.0e}.npz", active_synergies=active_synergies)

def save_synergies_mat(s_list: list[npt.NDArray], active: list[int], subject: str) -> None:
    """
    """
    subj_dir = get_subj_dir(NPZ_SYNERGIES_DIR, subject)

    # stack synergies into 3d array
    synergies = np.stack([s_list[j] for j in active], axis=2)
    # save in .mat
    mat_file = subj_dir / f"{subject}_synergies.mat"
    sio.savemat(mat_file, {"synergies": synergies})

    # confirmation
    print(f"Saved {len(active)} synergies to {mat_file} with shape {synergies.shape}")