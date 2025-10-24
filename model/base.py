import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional
from groupyr import SGL # type: ignore
from abc import ABC, abstractmethod

from sklearn import linear_model # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import sys

from model.utils.save_files import save_reconstruction_plot, save_synergy_plot, save_active_repeats

#np.set_printoptions(precision=4, threshold=sys.maxsize, suppress=True)
np.set_printoptions(precision=3, threshold=np.inf, linewidth=200)

SYNERGY_NORM_MAX = 4.0

class BaseSynergyModel(ABC):
    def __init__(self, T: int, t_s: int, m: int, n: int, K_j: int, G: int, subject: str, 
                 V: Optional[npt.NDArray] = None, seed: Optional[int] = None,
                 lambda1: float = 0.05, alpha: float = 0.05, epochs: int = 100):
        self.T = T  # duration of grasping task
        self.t_s = t_s  # duration of synergy
        self.m = m  # number of synergies
        self.m_true = 6
        self.n = n  # number of joints
        self.K_j = K_j  # number of repeats for each synergy
        self.G = G  # number of grasping tasks
        self.subject = subject  # the subject whose data we are using
        self.V: Optional[npt.NDArray] = V  # each column is a grasping task
        self.seed: Optional[int] = seed  # seed for init_synergies()
        self.lambda1 = lambda1  # combination between group lasso and lasso (0 is group lasso, 1 is lasso)
        self.alpha = alpha  # overall regularization strength
        self.epochs = epochs  # number of epochs for alternating minimization

        # Initialize true values of synergies and S and C (then solve for the true V)
        #self.true_s_list: list[npt.NDArray] = self.init_synergies(self.m_true)
        #self.true_S: npt.NDArray = self.build_S(self.true_s_list, self.m_true)
        #self.true_C: npt.NDArray = self.sparse_C()
        #self.V = self.V_est(self.true_S, self.true_C)

        # Initialize synergies and S and C
        # These are starting 'guesses' (in C's case, everything is initialized to 0)
        self.s_list: list[npt.NDArray] = self.init_synergies(self.m)
        self.S: npt.NDArray = self.build_S(self.s_list, self.m)
        self.C: npt.NDArray = self.init_C()

    def init_synergies(self, m: int) -> list[npt.NDArray]:
        """Initializes the synergies using random numbers.
        
        Params:
            None.
        Returns:
            list[npt.NDArray]: The list of synergy matrices.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        return [np.random.randn(self.n, self.t_s) for _ in range(m)]
    
    def shift_synergy(self, synergy: npt.NDArray, shift: int) -> npt.NDArray:
        """Shifts a synergy in time and stacks each joint vertically into a column vector.

        Params:
            synergy (npt.NDArray): Shape (num_joints, t_s), rows=joints, cols=timesteps.
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

    def build_S(self, synergies: list[npt.NDArray], m: int) -> npt.NDArray:
        """Builds the dictionary matrix S containing all shifted synergies.

        Params:
            synergies (list[npt.NDArray]): A list of synergy matrices, each of shape (num_joints, t_s)
            m (int): The number of synergies (length of synergies). This parameter is required only in
                     the testing phase because we have a different number of synergies for true_S and S.
        Returns:
            npt.NDArray: The S matrix, where each column is a shifted synergy, shape (nT, m * K_j).
        """
        cols = []  # holds the S columns

        # S contains all shifts of all synergies (K_j * m columns)
        # For each synergy, we shift it K_j, appending each shift to cols
        for j in range(m):
            s_j = synergies[j]
            for shift in range(self.K_j):
                col = self.shift_synergy(s_j, shift)
                cols.append(col)
        
        S = np.column_stack(cols)
        return S
    
    def update_S(self, synergies: list[npt.NDArray]) -> None:
        """Updates S with the newer/optimized synergies.
        
        Params:
            synergies (list[npt.NDArray]): A list of synergy matrices, each of shape (num_joints, t_s)
        Returns:
            None.
        """
        self.S = self.build_S(synergies, self.m)

    def solve_S(self) -> None:
        """Optimizes each synergy using the solve_s_j function.

        Params:
            None.
        Returns:
            None.
        """
        for j in range(self.m):
            #if not np.allclose(self.C[self.C_mask(j), :], 0):
            self.solve_s_j(j)
        #self.update_S(self.s_list)

    def solve_s_j(self, index: int) -> None:
        """Updates the synergy at s_list[index] using least squares with fixed C.

        Params:
            index (int): Index of the synergy in s_list.
        Returns:
            None.
        """
        # residuals for each grasping task with synergy j turned off: shape (nT, G)
        residual = self.calc_residual(index)
        # turn residual into column vector with grasping tasks stacked
        r_j = np.ravel(residual, order='F')

        B_blocks = []
        C_block = self.C[self.C_mask(index), :]

        for g in range(self.G):
            # coefficients corresponding to this grasping task, shape (K_j,)
            C_jg = C_block[:, g] 
            # build B_jg by combining shift matrices weighted by coefficients in C_jg
            B_jg = sum(
                C_jg[k] * self.build_shift_matrix(k) for k in range(self.K_j)
            )
            B_blocks.append(B_jg)

        # B_j is all grasping tasks stacked vertically, shape (nT * G, n * t_s)
        B_j = np.vstack(B_blocks)

        # optimize using least squares, reshape into (n, t_s), normalize
        #s_new, *_ = np.linalg.lstsq(B_j, r_j, rcond=None)
        #self.s_list[index] = np.reshape(s_new, (self.n, self.t_s))
        clf = linear_model.Ridge(alpha=0.25)
        clf.fit(B_j, r_j)
        self.s_list[index] = np.reshape(clf.coef_, (self.n, self.t_s))

        self.normalize_synergy(index)
        self.update_S(self.s_list)

    def build_shift_matrix(self, repeat: int) -> npt.NDArray:
        """Builds the shift matrix for the kth repeat of synergy j, shape (nT, nt_s).

        Params:
            repeat (int): The current repeat of synergy j.
        Returns:
            npt.NDArray: The shift matrix D_jk, shape (nT, nt_s).
        """
        # P_k is the shift matrix for one joint
        P_k = np.zeros(shape=(self.T, self.t_s))
        # compute the target row positions for the synergy after shifting by 'repeat'
        # [0, 1, 2, ..., t_s - 1] + k = [k, k + 1, k + 2, ..., k + t_s - 1]
        row_positions = np.arange(stop=self.t_s) + repeat

        # find which row positions are within the valid range of 0, ..., T - 1
        in_bounds = row_positions < self.T
        # insert ones into P_k at the positions corresponding to the shifted synergy
        P_k[row_positions[in_bounds], np.arange(stop=self.t_s)[in_bounds]] = 1.0

        D_jk = np.kron(np.eye(self.n), P_k)
        return D_jk

    def sparse_group_lasso(self) -> None:
        """Uses sparse group lasso to find the coefficients.

        Params:
            None.
        Returns:
            None.
        """
        indices = np.arange(self.m * self.K_j)
        groups = np.split(indices, self.m)

        col_norms = np.linalg.norm(self.S, axis=0) + 1e-12
        zero_cols = np.where(col_norms == 0)[0]
        if zero_cols.size > 0:
            col_norms[zero_cols] = 1.0
        S_scaled = self.S / col_norms[np.newaxis, :]

        for g in range(self.G):
            if self.V is None:
                raise ValueError("self.V cannot be None for sparse group lasso.")
            v_g = self.V[:, g]

            model = SGL(l1_ratio=self.lambda1, alpha=self.alpha, groups=groups)
            #model.fit(self.S, v_g)
            #self.C[:, g] = model.coef_
            model.fit(S_scaled, v_g)
            coef = model.coef_ / col_norms
            self.C[:, g] = coef

            # ---------------- Diagnostics ----------------
            nonzero = np.count_nonzero(np.abs(coef) > 1e-8)
            print(f"\nGrasp {g+1}: {nonzero} / {coef.size} nonzero coefficients")

            # show top 10 by magnitude
            top_idx = np.argsort(np.abs(coef))[::-1][:10]
            print("  Top |coef|:", np.round(np.abs(coef[top_idx]), 4))

            # per-group diagnostics
            for j in range(self.m):
                block = coef[self.C_mask(j)]
                nonzero_block = np.count_nonzero(np.abs(block) > 1e-8)
                max_block = np.max(np.abs(block))
                print(f"   Group {j+1}: {nonzero_block:2d}/{block.size} active, max={max_block:.3e}")

            # optional: check collinearity (just for first grasp to avoid clutter)
            if g == 0:
                j0 = 1  # choose which synergy to inspect (0-indexed)
                idx = self.C_mask(j0)
                S_block = self.S[:, idx]
                Szn = S_block / (np.linalg.norm(S_block, axis=0) + 1e-12)
                corrs = Szn.T @ Szn
                upper = corrs[np.triu_indices_from(corrs, k=1)]
                print(f"   Cosine sim (Group {j0+1}): mean={upper.mean():.3f}, max={upper.max():.3f}")

    def normalize_synergy(self, index: int) -> None:
        """Normalizes a synergy to remove scaling ambiguity.

        Params:
            index (int): Index of the synergy in s_list.
        Returns:
            None.
        """
        norm = np.linalg.norm(self.s_list[index].flatten())
        # we only normalize if the norm is greater than a certain value
        #if norm < 1e-12:
            #return
        if norm > SYNERGY_NORM_MAX:
            self.s_list[index] /= norm
            self.C[self.C_mask(index), :] *= norm

    def calc_residual(self, index: int) -> npt.NDArray:
        """Computes residuals with the contribution of the synergy at index removed.

        Params:
            index (int): Index of the synergy in s_list.
        Returns:
            npt.NDArray: Residual matrix where each column is the residual for a grasping task,
                         shape (nT, G).
        """
        temp_C = self.C.copy()
        temp_C[self.C_mask(index), :] = 0
        pred_minus_s = self.S @ temp_C
        return self.V - pred_minus_s

    def init_C(self) -> npt.NDArray:
        """Initializes C as a matrix where all entries are 0.
        
        Params:
            None.
        Returns:
            npt.NDArray: The C matrix, shape (m * K_j, G).
        """
        return np.zeros(shape=(self.m * self.K_j, self.G))

    def C_mask(self, index: int) -> slice:
        """Selects the rows of C that correspond to the selected synergy (includes all shifted versions).

        Params:
            index (int): Index of the synergy in s_list.
        Returns:
            slice: Slice object that selects the rows of C belonging to the selected synergy.
        """
        mask = slice(self.K_j * index, self.K_j * (index + 1))
        return mask

    def sparse_C(self, group_prob: float = 0.4, within_prob: float = 0.5) -> npt.NDArray:
        """
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        C_num_rows = self.true_S.shape[1]  # length of c is the number of columns of S
        C = np.zeros(shape=(C_num_rows, self.G))

        num_nonzero = ((self.m // 2) * self.K_j) // 4
        for g in range(self.G):
            nonzero_indices = np.random.choice(C_num_rows, size=num_nonzero, replace=False)
            col = np.zeros(C_num_rows)
            col[nonzero_indices] = np.random.randn(num_nonzero)
            C[:, g] = col
        return C
    
    def V_est(self, S: npt.NDArray, C: npt.NDArray) -> npt.NDArray:
        """Calculates the current V estimation using S and C.
        
        Params:
            S (npt.NDArray): Matrix of synergies and their repeats.
            C (npt.NDArray): Matrix of coefficients.
        Returns:
            npt.NDArray: The V estimation, where the columns are grasping tasks, shape (nT, G).
        """
        return S @ C

    def C_loss(self) -> float:
        return np.sum((self.true_C - self.C)**2)
    
    def S_loss(self) -> float:
        return np.sum((self.true_S - self.S)**2)
    
    def V_grasp_loss(self, grasp: int) -> float:
        # get the column of V that corresponds to grasp
        V_col = self.V[:, grasp]
        V_est_col = self.V_est(self.S, self.C)[:, grasp]

        grasp_loss = np.sum((V_col - V_est_col)**2) / np.sum(V_col**2)
        return grasp_loss
    
    def V_loss(self) -> float:
        """Uses the same formula in self.V_grasp_loss to calculate the total V loss.

        Params:
            None.
        Returns:
            float: The loss (difference in actual and predicted value of V and V_est, respectively, squared).
        """
        V_est = self.V_est(self.S, self.C)
        loss = np.sum((self.V - V_est)**2) / np.sum(self.V**2)
        return loss

    def compare_V(self, compare: npt.NDArray, epoch: int) -> None:
        """Comparison of V and estimated V (per grasp).

        Params:
            compare (npt.NDArray): Numpy array to compare with V.
            epoch (int): Current epoch for alternating minimization.
        Returns:
            None.
        """
        if self.V is None:
            raise ValueError("self.V cannot be None for comparison.")
        
        V_est = self.V_est(self.S, self.C)
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
                fig.suptitle(f"Epoch {epoch}, Grasp {g + 1}, reconstruction error: {self.V_grasp_loss(g):.4f}")
                
                save_reconstruction_plot(fig, g, self.subject)
                plt.show()
                plt.close(fig)
                
        except KeyboardInterrupt:
            print("Keyboard Interrupt - skipping rest of loop.")

    def plot_synergies(self) -> None:
        """
        """
        for j in range(self.m):
            fig, axes = plt.subplots(self.n, 1, figsize=(6, 8), sharex=True)
            # normalize axes to always be a list
            if isinstance(axes, np.ndarray):
                axes_list: list[Axes] = axes.ravel().tolist()
            else:
                axes_list: list[Axes] = [axes]

            for i in range(self.n):
                ax: Axes = axes_list[i]
                joint = self.s_list[j][i, :]

                # compute y-axis numbers to display
                max_val = np.abs(joint).max()
                epsilon = 0.01
                ax.set_ylim(-max_val - epsilon, max_val + epsilon)
                ax.set_xlim(right=40)

                ax.plot(joint, 'g', label="True V")

                if i == self.n - 1:
                    ax.set_xlabel("Samples")
                if i == int(self.n / 2):
                    ax.set_ylabel("Angular velocities of ten joints (radian/sample)")
                    ax.set_xticks([0, 10, 20, 30, 40])

            plt.tight_layout(rect=[0, 0, 1, 0.95])       
            fig.suptitle(f"Synergy {j + 1}")
            save_synergy_plot(fig, j, self.subject)
            plt.show()
            plt.close(fig)
    
    def active_synergies(self, tol: float = 1e-6) -> tuple[list[int], list[int], npt.NDArray]:
        """
        """
        active: list[int] = [] 
        dropped: list[int] = []
        group_norms = np.zeros(self.m)

        for j in range(self.m):
            block = self.C[self.C_mask(j), :]
            group_norm = np.linalg.norm(block, ord='fro')
            group_norms[j] = group_norm
            (active if group_norm > tol else dropped).append(j)
        return active, dropped, group_norms
    
    def save_active_synergies(self, tol: float = 1e-6) -> npt.NDArray:
        """Computes and saves the active shifts of the active synergies.

        Params:
            tol (float): Threshold that determines which coefficients in C are considered inactive.
        Returns:
            npt.NDArray: The reduced S matrix that contains only active shifts.
        """
        active, *_ = self.active_synergies(tol)
        active_cols = []

        for j in active:
            C_block = self.C[self.C_mask(j), :]  # shape (K_j, G)
            S_cols = self.S[:, self.C_mask(j)]  # corresponding S cols

            # find shifts that have any nonzero coefficient across any grasp
            print(f"Synergy {j+1}: C_block shape={C_block.shape}, max|coef|={np.max(np.abs(C_block)):.3e}")
            print("Row max abs:", np.round(np.max(np.abs(C_block), axis=1), 5))

            active_shifts = np.any(np.abs(C_block) > tol, axis=1)
            active_cols.append(S_cols[:, active_shifts])
            print(f"Number of active repeats for synergy {j + 1}: {np.sum(active_shifts)}")
        
        active_S = np.column_stack(active_cols)
        save_active_repeats(active_S, self.subject, tol)
        print(active_S.shape)
        return active_S
    
    def surviving_synergy_density(self, active_synergies: list[int]) -> None:
        for j in active_synergies:
            block = self.C[self.C_mask(j), :]
            zero = np.count_nonzero(np.abs(block) < 1e-4)
            total = block.size
            print(f"Synergy {j}: {zero} / {total} zero ({zero / total :.1%})")

    def print_synergy_norms(self) -> None:
        """Prints the Frobenius norm of each synergy for diagnostic purposes."""
        norms = [np.linalg.norm(s) for s in self.s_list]
        print("\nSynergy norms:")
        for j, norm in enumerate(norms):
            print(f"  Synergy {j+1:2d}: {norm:.6f}")
        print(f"  Mean norm: {np.mean(norms):.6f}, Std: {np.std(norms):.6f}")

    @abstractmethod
    def solve(self) -> None:
        """
        """
        pass