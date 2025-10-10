import os
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from pathlib import Path

def get_root_path() -> Path:
    """
    """
    root = os.getenv("DATA_ROOT")
    if not root:
        raise RuntimeError("DATA_ROOT not set. Was load_dotenv() ever called in main.py?")
    return Path(root)

def load_mat_file(subject: str, filename: str) -> tuple[int, int, int, npt.NDArray]:
    """Loads a .mat file and returns the data.

    Params:
        subject (str): Which testing subject's data we want to examine.
        filename (str): The name of the file containing the subject's testing data.
    Returns:
        npt.NDArray: The joint angular velocity data from the .mat file.
    """
    root_path = get_root_path()
    file_path = root_path / subject / filename
    data_dict = sio.loadmat(file_path)
    data: npt.NDArray = data_dict['testdata']
    n, T, G = data.shape  # get number of joints, timesteps, and grasping tasks
    V = data.reshape(n * T, G)  # new V matrix
    return n, T, G, V

def load_natural_grasps(subject: str, filename: str) -> tuple[int, int, int, npt.NDArray]:
    """Loads the natural grasping tasks for a subject.
    """
    n, T, G, V = load_mat_file(subject, filename)
    return n, T, G, V

def load_asl(subject: str, filename: str) -> tuple[int, int, int, npt.NDArray]:
    """Loads the ASL tasks for a subject.
    """
    n, T, G, V = load_mat_file(subject, filename)
    return n, T, G, V

def get_npz(filename: str) -> npt.NDArray:
    """
    """
    curr_dir = Path(__file__).resolve().parent
    npz_path = curr_dir / "results" / filename
    data = np.load(npz_path)
    return data['active_synergies']