import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import Lasso
from pathlib import Path
from dotenv import load_dotenv

from model.utils.read_files import load_asl, get_npz
from model.utils.save_files import save_asl_plot

load_dotenv()

class TestASL():
    def __init__(self, subject: str, asl_filename: str, npz_filename: str):
        self.subject = subject
        self.n, self.T, self.G, self.V = load_asl(self.subject, asl_filename)
        self.S_bank = get_npz(subject, npz_filename)
        self.C = np.zeros(shape=(self.S_bank.shape[1], self.G))

    def find_coeff(self) -> None:
        for g in range(self.G):
            v_g = self.V[:, g]
            #self.C[:, g], *_ = np.linalg.lstsq(self.S_bank, v_g, rcond=None)
            lasso = Lasso(alpha=0.000075)
            lasso.fit(self.S_bank, v_g)
            self.C[:, g] = lasso.coef_

            tol = 1e-4
            mask = np.abs(lasso.coef_) > tol
            count = np.sum(mask)
            print(f"Number of selected columns of S (tol = {tol}): {count}")

    def grasp_loss(self, grasp: int) -> float:
        # get the column of V that corresponds to grasp
        V_col = self.V[:, grasp]
        V_est_col = self.V_est()[:, grasp]

        grasp_loss = np.sum((V_col - V_est_col)**2) / np.sum(V_col**2)
        return grasp_loss
    
    def V_loss(self) -> float:
        """Uses the same formula in self.grasp_loss to calculate the total V loss.

        Params:
            None.
        Returns:
            float: The loss (difference in actual and predicted value of V and V_est, respectively, squared).
        """
        V_est = self.V_est()
        loss = np.sum((self.V - V_est)**2) / np.sum(self.V**2)
        return loss

    def V_est(self) -> npt.NDArray:
        """
        """
        return self.S_bank @ self.C

    def asl_reconstruction_plot(self) -> None:
        """
        """
        V_est = self.V_est()
        try:
            for g in range(self.G):
                fig, axes = plt.subplots(self.n, 1, figsize=(6, 8), sharex=True)
                # normalize axes to always be a list
                if isinstance(axes, np.ndarray):
                    axes_list: list[Axes] = axes.ravel().tolist()
                else:
                    axes_list: list[Axes] = [axes]

                for i in range(self.n):
                    ax: Axes = axes_list[i]
                    true_v = self.V[(i * self.T):((i + 1) * self.T), g]
                    est_v = V_est[(i * self.T):((i + 1) * self.T), g]

                    # compute y-axis numbers to display
                    max_val = max(np.abs(true_v).max(), np.abs(est_v).max())
                    ax.set_ylim(-max_val, max_val)

                    ax.plot(true_v, 'r', label="True V")
                    ax.plot(est_v, 'k', label="Estimate V")

                    if i == self.n - 1:
                        ax.set_xlabel("Samples")
                    if i == int(self.n / 2):
                        ax.set_ylabel("Angular velocities of ten joints (radian/sample)")
                        ax.set_xticks([0, 20, 40, 60, 80])

                handles, labels = axes_list[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", fontsize=8)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fig.suptitle(f"Grasp {g + 1}, ASL reconstruction error: {self.grasp_loss(g):.4f}")

                save_asl_plot(fig, g, self.subject)
                plt.show()
                plt.close(fig)

        except KeyboardInterrupt:
                print("Keyboard Interrupt - skipping rest of loop.")
    
    def run(self) -> None:
        self.find_coeff()
        print(f"V loss {self.V_loss()}")
        self.asl_reconstruction_plot()

if __name__ == '__main__':
    subject = "subj2"
    asl_filename = "ASL_Test_Data.mat"
    npz_filename = "active_synergies_tol=1e-04.npz"
    asl = TestASL(subject, asl_filename, npz_filename)
    # run
    print(asl.S_bank.shape)
    asl.run()