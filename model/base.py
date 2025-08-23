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

        # initialize synergies and S and C
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

    def init_S(self) -> npt.NDArray:
        return 8

    def init_C(self) -> npt.NDArray:
        return np.zeros(shape=(self.m * self.K_j, self.G))
    
    def V_est(self) -> npt.NDArray:
        """Calculates the current V estimation using S and C.
        
        Params:
            None.
        Returns:
            npt.NDArray: The V estimation (cols are grasping tasks).
        """
        return self.S @ self.C
    
    """
    Uses squared L2 norm to calculate V loss

    Paramters:
        None
    Returns:
        Total loss
    """
    def V_loss(self) -> float:
        return np.sum((self.V - self.V_est())**2)

    @abstractmethod
    def solve(self, *args, **kwargs) -> None:
        pass