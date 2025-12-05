from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import scipy.io as sio
import numpy as np
import numpy.typing as npt

THIS_DIR = Path(__file__).resolve().parent

RESULTS_DIR = THIS_DIR / "results"
ASL_RECONSTRUCTION_MAT_DIR = RESULTS_DIR / "asl_reconstruction_mat"
NPZ_SYNERGIES_DIR = RESULTS_DIR / "npz_saved_synergies"
IMAGES_DIR = RESULTS_DIR / "images"
SYNERGY_DIR = IMAGES_DIR / "synergies"
RECON_DIR = IMAGES_DIR / "reconstruction"
ASL_DIR = IMAGES_DIR / "asl_reconstruction"

# build folders if they do not already exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASL_RECONSTRUCTION_MAT_DIR.mkdir(parents=True, exist_ok=True)
NPZ_SYNERGIES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SYNERGY_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR.mkdir(parents=True, exist_ok=True)
ASL_DIR.mkdir(parents=True, exist_ok=True)

def get_subj_dir(base_dir: Path, subject: str) -> Path:
    """Creates a subject directory inside the base directory (if folder DNE).
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
    #print(f"Saved {len(active)} synergies to {mat_file} with shape {synergies.shape}")

def save_grasp_mat(v_data: npt.NDArray, filename: str, subject: str) -> None:
    """
    """
    subj_dir = get_subj_dir(ASL_RECONSTRUCTION_MAT_DIR, subject)

    # save in .mat
    mat_file = subj_dir / filename
    sio.savemat(mat_file, {"velocities": v_data})

    # confirmation
    #print(f"Saved velocities to {mat_file} with shape {v_data.shape}")

def save_synergy_graphs_together(subject: str) -> None:
    """
    """
    subj_dir = get_subj_dir(NPZ_SYNERGIES_DIR, subject)
    path = subj_dir / f"{subject}_synergies.mat"
    file = sio.loadmat(path)
    synergies = file["synergies"]
    n, T_s, m = synergies.shape
    num_m = 3

    fig, axes = plt.subplots(n, 3, figsize=(12, 8), sharex=True)
    # axes must be 2d
    if n == 1:
        axes = axes[np.newaxis, :]
    if m == 1:
        axes = axes[:, np.newaxis]

    joints = ["T-MCP", "T-IP", "I-MCP", "I-PIP", "M-MCP", "M-PIP", "R-MCP", "R-PIP", "P-MCP", "P-PIP"]
    for j in range(num_m):
        for i in range(n):
            ax: Axes = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.xaxis.set_major_locator(MultipleLocator(-(T_s // -4)))
            joint = synergies[i, :, j]

            # plot
            ax.plot(joint, color="g")
            max_val = np.abs(joint).max()
            tick_val = np.ceil(max_val * 100) / 100  # rounds up to 0.01
            y_plus_lim = tick_val + 0.01
            y_minus_lim = -tick_val - 0.01
            ax.set_ylim([y_minus_lim, y_plus_lim])
            ax.set_xlim([0, T_s + 1])

            ax.set_yticks([y_minus_lim, 0, y_plus_lim])
            
            if j == num_m - 1:
                ax.set_ylabel(f"{joints[i]}", labelpad=40, rotation="horizontal", fontsize=18)
                ax.yaxis.set_label_position("right")
            if i == n - 1:
                ax.set_xlabel("Samples", fontsize=18)
        
        axes[0, j].set_title(f"Synergy {j + 1}", fontsize=18)
    
    fig.text(
        0.01, 0.5, "Angular velocities of ten joints (radian/sample)",
        va="center", rotation="vertical", fontsize=18
    )
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    plt.savefig(subj_dir / "three_synergies.png")
    plt.show()

if __name__ == '__main__':
    save_synergy_graphs_together(subject="subj2")