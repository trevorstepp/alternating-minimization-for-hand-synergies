import numpy as np
import numpy.typing as npt
from typing import Optional
from sklearn import linear_model # type: ignore
from asgl import Regressor # type: ignore

from model.base import BaseSynergyModel

class AlternatingMinModel(BaseSynergyModel):
    def __init__(self, T: int, t_s: int, m: int, n: int, K_j: int, G: int, V: Optional[npt.NDArray] = None, seed: Optional[int] = None, epochs: int = 100):
        super().__init__(T, t_s, m, n, K_j, G, V, seed)
        self.epochs = epochs