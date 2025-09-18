import os
import numpy as np
import numpy.typing as npt
import scipy.io as sio
from pathlib import Path

#subject = "subj1"
#filename = "Natural_Test_Data.mat"

def load_mat_data(subject: str, filename: str) -> tuple[int, int, int, npt.NDArray]:

    root_path = Path(os.getenv("DATA_ROOT"))
    file_path = root_path / subject / filename
    data_dict = sio.loadmat(file_path)
    data: npt.NDArray = data_dict['testdata']
    #print(data)

    n, T, G = data.shape  # get number of joints, timesteps, and grasping tasks
    V = data.reshape(n * T, G)  # new V matrix

    return n, T, G, V