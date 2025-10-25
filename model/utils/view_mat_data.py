import os
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
    print(data_dict)
    #data: npt.NDArray = data_dict['testdata']
    """
    n, T, G = data.shape  # get number of joints, timesteps, and grasping tasks
    V = data.reshape(n * T, G)  # new V matrix
    return n, T, G, V
    """

def load_asl(subject: str, filename: str) -> tuple[int, int, int, npt.NDArray]:
    """Loads the ASL tasks for a subject.
    """
    n, T, G, V = load_mat_file(subject, filename)
    return n, T, G, V

if __name__ == '__main__':
    subj = "subj2"
    filename = "ASL_Test_Data.mat"
    load_mat_file(subj, filename)