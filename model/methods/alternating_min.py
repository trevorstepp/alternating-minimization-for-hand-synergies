import numpy as np
from sklearn import linear_model # type: ignore
from asgl import Regressor # type: ignore

from model.base import BaseSynergyModel

class AlternatingMinModel(BaseSynergyModel):

    def solve(self, epochs: int=100) -> None:
        pass