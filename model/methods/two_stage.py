import numpy as np
import numpy.typing as npt
from typing import Optional
from sklearn import linear_model # type: ignore

from model.base import BaseSynergyModel

class TwoStage(BaseSynergyModel):
    def __init__(self, T: int, t_s: int, m: int, n: int, K_j: int, G: int, V: Optional[npt.NDArray] = None, 
                 seed: Optional[int] = None, lambda1: float = 0.05, alpha: float = 0.05, epochs: int = 100, lasso_param: float = 0.01):
        super().__init__(T, t_s, m, n, K_j, G, V, seed, lambda1, alpha, epochs)
        self.lasso_param = lasso_param

    def lasso(self) -> None:
        """Uses Lasso regression to produce estimates for coefficients.

        """
        # Lasso regression
        lasso = linear_model.Lasso(alpha=self.lasso_param, fit_intercept=False)
        for g in range(self.G):
            lasso.fit(self.S, self.V[:, g])
            self.C[:, g] = lasso.coef_

    def solve(self) -> None:
        """
        
        """
        for epoch in range(self.epochs):
            self.lasso()
            self.solve_S()
            print(f"Epoch {epoch}")
            #print(f"S loss: {self.S_loss()}")
            #print(f"C loss: {self.C_loss()}")
            print(f"V loss {self.V_loss()}")
            if epoch % 10 == 0:
                self.compare_V(self.V_est(self.true_S, self.true_C), epoch)

        print("Before sparse group lasso:")
        print(f"{self.C}")
        active, dropped, group_norms = self.active_synergies()
        print(f"Active: {active}; Dropped: {dropped}")

        print("After sparse group lasso:")
        self.sparse_group_lasso()
        print(f"{self.C}")
        active, dropped, group_norms = self.active_synergies()
        print(f"Active: {active}; Dropped: {dropped}")
        print(f"V loss {self.V_loss()}")
        self.compare_V(self.V_est(self.true_S, self.true_C), epoch)