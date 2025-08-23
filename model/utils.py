# utils.py holds helper functions

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def shift_synergy(self, synergy: npt.NDArray, shift: int) -> npt.NDArray:
    """Shifts a synergy in time and stacks each joint vertically into a column vector.

    Params:
        synergy (npt.NDArray): A matrix whose rows are joints and whose cols are timestamps.
        shift (int): The amount by which to shift the synergy.
    Returns:
        npt.NDArray: The column vector that holds the shifted synergy (all joints stacked).
    """
    shifts = []  # to hold shifted rows of s
    for j in range(self.num_joints):
        front_pad = np.zeros(shift)
        back_pad = np.zeros(self.T - self.t_s - shift)
        shift_j = np.concatenate([front_pad, synergy[j], back_pad])
        shifts.append(shift_j)
    return np.concatenate(shifts)  # shape (num_joints * T,)