from scipy.signal import savgol_filter

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.linear_model import Lasso
from pathlib import Path
from dotenv import load_dotenv

from model.utils.read_files import load_asl, get_npz
from model.utils.save_files import save_asl_plot, save_grasp_mat

load_dotenv()

class TestASL():
    def __init__(self, subject: str, asl_filename: str, npz_filename: str, alpha: float):
        self.subject = subject
        self.alpha = alpha
        self.n, self.T, self.G, self.V = load_asl(self.subject, asl_filename)

        self.K_j = 44
        self.m = 7
        self.S_bank_t = get_npz(subject, npz_filename)
        self.labels_t = self.create_labels(self.S_bank_t)

        self.S_bank, self.labels, self.kept = self.prune_correlated(self.S_bank_t, self.labels_t, threshold=0.8)
        self.S_bank = savgol_filter(self.S_bank, window_length=11, polyorder=3, axis=0)
        assert self.S_bank.shape[1] == self.labels.shape[0]
        print(f"Kept {len(self.kept)} out of {self.S_bank_t.shape[1]} columns after correlation pruning.")
        self.C = np.zeros(shape=(self.S_bank.shape[1], self.G))

    def create_labels(self, S: npt.NDArray) -> npt.NDArray:
        labels = []
        for j in range(self.m):
            for k in range(self.K_j):
                labels.append((j, k))  # (synergy j, shift k)
        labels = np.array(labels)  # shape (M, 2), M = self.m * self.K_j
        return labels

    def prune_correlated(self, S: npt.NDArray, labels: npt.NDArray, threshold: float=0.9) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
        """
        """
        corr_matrix = np.corrcoef(S, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)
        to_remove = set()

        for i in range(corr_matrix.shape[0]):
            if i in to_remove:
                continue
            high_corr = np.where(np.abs(corr_matrix[i]) > threshold)[0]
            to_remove.update(high_corr)

        keep_indices = [i for i in range(S.shape[1]) if i not in to_remove]
        return S[:, keep_indices], labels[keep_indices], keep_indices

    def lasso(self) -> None:
        tol=1e-3
        all_losses = []
        all_counts = []
        for g in range(self.G):
            alpha = self.alpha
            v_g = self.V[:, g]
            v_norm = np.linalg.norm(v_g)

            while True:
                lasso = Lasso(alpha=alpha, max_iter=1000)
                lasso.fit(self.S_bank, v_g)
                self.C[:, g] = lasso.coef_
                mask = np.abs(self.C[:, g]) > tol
                count = np.sum(mask)
                """
                if(((count < 8 and self.grasp_loss(g) > 0.15) or self.grasp_loss(g) > 0.25) and count <= 15):
                    alpha /= 1.05
                    #print(f"{g + 1}: Repeat")
                else:
                    #print(f"{g + 1}: Done")
                    break
                """
                break

            mask = np.abs(self.C[:, g]) > tol
            count = np.sum(mask)
            loss = self.grasp_loss(g)
            all_losses.append(loss)
            all_counts.append(count)
            print(f"Grasp {g + 1}: Number of selected columns of S (tol = {tol}): {count}, v_norm = {v_norm}, loss = {self.grasp_loss(g):.4f}")

        total_loss = np.mean(all_losses)
        total_sparsity = np.mean(all_counts)
        print("\n===== ASL Reconstruction Diagnostics =====")
        print(f"Mean per-grasp loss: {total_loss:.4f}")
        print(f"Mean number of selected synergies: {total_sparsity:.2f}")
        print(f"Total V-loss (across all grasps): {self.V_loss():.4f}")

        # Per-synergy usage (how often each synergy shift was active)
        usage_counts = np.sum(np.abs(self.C) > tol, axis=1)
        top_used = np.argsort(-usage_counts)[:10]  # top 10 active columns
        print(f"Most used synergy shifts (indices): {top_used}")
        print(f"Usage counts of top shifts: {usage_counts[top_used]}")
    
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
            joints = ["T-MCP", "T-IP", "I-MCP", "I-PIP", "M-MCP", "M-PIP", "R-MCP", "R-PIP", "P-MCP", "P-PIP"]
            for g in range(self.G):
                fig, axes = plt.subplots(self.n, 1, figsize=(8, 10), sharex=True)
                # normalize axes to always be a list
                if isinstance(axes, np.ndarray):
                    axes_list: list[Axes] = axes.ravel().tolist()
                else:
                    axes_list: list[Axes] = [axes]

                for i in range(self.n):
                    ax: Axes = axes_list[i]
                    ax.tick_params(axis='both', which='major', labelsize=12)

                    true_v = self.V[(i * self.T):((i + 1) * self.T), g]
                    est_v = V_est[(i * self.T):((i + 1) * self.T), g]

                    # compute y-axis numbers to display
                    max_val = max(np.abs(true_v).max(), np.abs(est_v).max())
                    tick_val = np.ceil(max_val * 100) / 100  # rounds up to 0.01
                    y_plus_lim = tick_val + 0.01
                    y_minus_lim = -tick_val - 0.01
                    ax.set_ylim([y_minus_lim, y_plus_lim])
                    ax.set_yticks([y_minus_lim, 0, y_plus_lim])

                    ax.plot(true_v, 'r', label="True V")
                    ax.plot(est_v, 'k', label="Estimate V")

                    if i == self.n - 1:
                        ax.set_xlabel("Samples", fontsize=18)
                        ax.set_xticks([0, 20, 40, 60, 80])
                    
                    ax.set_ylabel(f"{joints[i]}", labelpad=40, rotation="horizontal", fontsize=18)
                    ax.yaxis.set_label_position("right")
                
                fig.text(
                    0.01, 0.5, "Angular velocities of ten joints (radian/sample)",
                    va="center", rotation="vertical", fontsize=18
                )
                fig.suptitle(f"Gesture {g + 1}, ASL reconstruction error: {self.grasp_loss(g):.4f}", fontsize=18)
                plt.tight_layout(rect=(0.025, 0, 1, 1))

                # save plots and values
                save_asl_plot(fig, g, self.subject)
                # need to reshape velocities
                true_grasp_matrix = self.V[:, g].reshape(self.n, self.T)
                recon_grasp_matrix = V_est[:, g].reshape(self.n, self.T)
                #save_grasp_mat(true_grasp_matrix, f"{self.subject}_ASL{g + 1}.mat", self.subject)
                #save_grasp_mat(recon_grasp_matrix, f"{self.subject}_ASL{g + 1}_reconstruction.mat", self.subject)
                #plt.show()
                plt.close(fig)

        except KeyboardInterrupt:
                print("Keyboard Interrupt - skipping rest of loop.")

    def plot_shift_usage(self, grasp: int) -> None:
        tol = 1e-3
        # select active coefficients
        mask = np.abs(self.C[:, grasp]) > tol
        active = np.flatnonzero(mask)

        fig, axes = plt.subplots(self.m, 1, figsize=(8, 6), sharex=True)
        for index in active:
            synergy, shift = self.labels[index]
            amplitude = abs(self.C[index, grasp])

            # select the subplot
            ax: Axes = axes[synergy]

            # scatter plot
            ax.stem([shift], [amplitude], linefmt='purple', markerfmt='o', basefmt=' ')
        
        fig.text(
            0.01, 0.5, "Amplitude of shifts",
            va="center", rotation="vertical", fontsize=18
        )
        axes[-1].set_xlabel("Time of recruitment (samples)", fontsize=18)
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)
        fig.suptitle(f"ASL Gesture {grasp + 1}: Synergy Recruitment", fontsize=18)
        plt.tight_layout(rect=(0.025, 0, 1, 1))
        plt.show()
    
    def run(self) -> None:
        self.lasso()
        #print(f"V loss {self.V_loss()}")
        #self.asl_reconstruction_plot()
        self.plot_shift_usage(grasp=28)

if __name__ == '__main__':
    subject = "subj8"
    asl_filename = "ASL_Test_Data.mat"
    npz_filename = "active_synergies_tol=1e-04.npz"
    asl = TestASL(subject, asl_filename, npz_filename, alpha=0.00002)
    # run
    print(asl.S_bank.shape)
    asl.run()