from model.base import BaseSynergyModel

class AlternatingMinModel(BaseSynergyModel):
    def solve(self) -> None:
        """
        
        """
        for epoch in range(self.epochs):
            self.sparse_group_lasso()
            self.solve_S()
            print(f"Epoch {epoch}")
            #print(f"S loss: {self.S_loss()}")
            #print(f"C loss: {self.C_loss()}")
            print(f"V loss {self.V_loss()}")
            if epoch % 10 == 0:
                self.compare_V(self.V_est(self.true_S, self.true_C), epoch)

        self.compare_V(self.V_est(self.true_S, self.true_C), epoch)  
        active, dropped, group_norms = self.active_synergies()
        print(f"Active: {active}; Dropped: {dropped}")
        print(f"{self.C}")