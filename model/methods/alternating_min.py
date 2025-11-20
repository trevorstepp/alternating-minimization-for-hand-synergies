from model.base import BaseSynergyModel
import numpy as np
from model.utils.read_files import verify_saved_synergies

class AlternatingMinModel(BaseSynergyModel):
    def solve(self) -> None:
        """
        
        """
        try:
            print(f"V loss {self.V_loss()}")
            for epoch in range(self.epochs):
                self.sparse_group_lasso()
                self.solve_S()
                print(f"Epoch {epoch + 1}")
                #print(f"S loss: {self.S_loss()}")
                #print(f"C loss: {self.C_loss()}")
                print(f"V loss {self.V_loss()}")

                if (epoch + 1) % 10 == 0:
                    cont = input("Continue loop (y or n)? ")
                    if cont == 'n':
                        break
                #if epoch % 10 == 0:
                    #self.compare_V(self.V_est(self.true_S, self.true_C), epoch)
                    #self.compare_V(self.V_est(self.S, self.C), epoch)
        except KeyboardInterrupt:
            print("Keyboard Interrupt - skipping rest of loop.")

        #self.compare_V(self.V_est(self.true_S, self.true_C), epoch) 
        self.compare_V(self.V_est(self.S, self.C), epoch + 1) 
        save = input("Save synergies (y or n)? ")
        if(save == 'y'):
            self.plot_synergies()

        #self.print_synergy_norms()
        self.save_active_synergies(tol=1e-4)

        active, dropped, group_norms = self.active_synergies()
        print(f"Active: {active}; Dropped: {dropped}")
        print(f"Group norms: {group_norms}")
        print(f"Number of zero coefficients in C: {np.count_nonzero(np.abs(self.C) < 1e-8)} / {np.size(self.C)}")

        expected_synergies = np.stack([self.s_list[j] for j in active], axis=2)
        saved_synergies = verify_saved_synergies(self.subject, f"subj{self.subject}_synergies.mat")

        print("Saved synergies shape:", saved_synergies.shape)
        print("Expected synergies shape:", expected_synergies.shape)

        if np.array_equal(expected_synergies, saved_synergies):
            print("SUCCESS: saved synergies match")