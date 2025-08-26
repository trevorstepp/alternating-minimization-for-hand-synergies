import numpy as np
import numpy.typing as npt
from typing import Optional
from abc import ABC, abstractmethod

class BaseSynergyModel(ABC):
    def __init__(self, T: int, t_s: int, m: int, n: int, K_j: int, G: int, V: Optional[npt.NDArray]=None, seed: Optional[int]=None):
        self.T = T  # duration of grasping task
        self.t_s = t_s  # duration of synergy
        self.m = m  # number of synergies
        self.n = n  # number of joints
        self.K_j = K_j  # number of repeats for each synergy
        self.G = G  # number of grasping tasks
        self.V: Optional[npt.NDArray] = V  # each column is a grasping task
        self.seed: Optional[int] = seed  # seed for init_synergies()

        # Initialize synergies and S and C
        self.s_list: list[npt.NDArray] = self.init_synergies()
        self.S: npt.NDArray = self.init_S()
        self.C: npt.NDArray = self.init_C()

    def init_synergies(self) -> list[npt.NDArray]:
        """Initializes the synergies using random numbers.
        
        Params:
            None.
        Returns:
            list[npt.NDArray]: The list of synergy matrices.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        return [np.random.randn(self.n, self.t_s) for _ in range(self.m)]
    
    def shift_synergy(self, synergy: npt.NDArray, shift: int) -> npt.NDArray:
        """Shifts a synergy in time and stacks each joint vertically into a column vector.

        Params:
            synergy (npt.NDArray): shape (num_joints, t_s), rows=joints, cols=timesteps.
            shift (int): The amount by which to shift the synergy.
        Returns:
            npt.NDArray: The column vector that holds the shifted synergy (all joints stacked).
        """
        shifts = []  # holds shifted rows of s
        for j in range(self.n):
            front_pad = np.zeros(shift)
            back_pad = np.zeros(self.T - self.t_s - shift)
            shift_j = np.concatenate([front_pad, synergy[j], back_pad])
            shifts.append(shift_j)
        return np.concatenate(shifts)  # shape (num_joints * T,)

    def build_S(self) -> npt.NDArray:
        """Builds the dictionary matrix S containing all shifted synergies.

        Params:
            None.
        Returns:
            npt.NDArray: The S matrix, where each column is a shifted synergy, shape (nT, m * K_j).
        """
        cols = []  # holds the S columns

        # S contains all shifts of all synergies (K_j * m columns)
        # For each synergy, we shift it K_j, appending each shift to cols
        for j in range(self.m):
            s_j = self.s_list[j]
            for shift in range(self.K_j):
                col = self.shift_synergy(s_j, shift)
                cols.append(col)
        
        S = np.column_stack(cols)
        return S

    def init_C(self) -> npt.NDArray:
        """Initializes C as a matrix where all entries are 0.
        
        Params:
            None.
        Returns:
            npt.NDArray: The C matrix, shape (m * K_j, G).
        """
        return np.zeros(shape=(self.m * self.K_j, self.G))
    
    def V_est(self) -> npt.NDArray:
        """Calculates the current V estimation using S and C.
        
        Params:
            None.
        Returns:
            npt.NDArray: The V estimation, where the columns are grasping tasks, shape (nT, G).
        """
        return self.S @ self.C
    
    def V_loss(self) -> float:
        """Uses squared L2 norm to calculate V loss.

        Params:
            None.
        Returns:
            float: The loss (difference in actual and predicted value of V and V_est, respectively, squared).
        """
        return np.sum((self.V - self.V_est())**2)

    @abstractmethod
    def solve(self, *args, **kwargs) -> None:
        pass