import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from groupyr import SGL
from abc import ABC, abstractmethod

from sklearn import linear_model # type: ignore
from sklearn.preprocessing import StandardScaler
import sys

np.set_printoptions(precision=4, threshold=sys.maxsize, suppress=True)

SYNERGY_NORM_MAX = 4.0

class BaseSynergyModel(ABC):
    def __init__(self, T: int, t_s: int, m: int, n: int, K_j: int, G: int, 
                 V: Optional[npt.NDArray] = None, seed: Optional[int] = None,
                 lambda1: float = 0.05, alpha: float = 0.05, epochs: int = 100):
        self.T = T  # duration of grasping task
        self.t_s = t_s  # duration of synergy
        self.m = m  # number of synergies
        self.m_true = m // 2
        self.n = n  # number of joints
        self.K_j = K_j  # number of repeats for each synergy
        self.G = G  # number of grasping tasks
        self.V: Optional[npt.NDArray] = V  # each column is a grasping task
        self.seed: Optional[int] = seed  # seed for init_synergies()
        self.lambda1 = lambda1  # combination between group lasso and lasso (0 is group lasso, 1 is lasso)
        self.alpha = alpha  # overall regularization strength
        self.epochs = epochs  # number of epochs for alternating minimization

        # Initialize true values of synergies and S and C (then solve for the true V)
        self.true_s_list: list[npt.NDArray] = self.init_synergies(self.m_true)
        self.true_S: npt.NDArray = self.build_S(self.true_s_list, self.m_true)
        self.true_C: npt.NDArray = self.sparse_C()
        self.V = self.V_est(self.true_S, self.true_C)

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

    def build_S(self, synergies: list[npt.NDArray], m: int) -> npt.NDArray:
        """Builds the dictionary matrix S containing all shifted synergies.

        Params:
            None.
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
    
    def update_S(self, synergies: npt.NDArray) -> None:
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
        s_new, *_ = np.linalg.lstsq(B_j, r_j, rcond=None)
        self.s_list[index] = np.reshape(s_new, (self.n, self.t_s))
        self.normalize_synergy(index)
        self.update_S(self.s_list)

    def build_shift_matrix(self, repeat: int) -> npt.NDArray:
        """Builds the shift matrix for synergy j, repeat k, shape (nT, nt_s)

        Params:
            repeat (int): the current repeat (k) of synergy j
        Returns:
            npt.NDArray:
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
        indices = np.arange(self.m * self.K_j)
        groups = np.split(indices, self.m)

        scaler = StandardScaler()
        S_scaled = scaler.fit_transform(self.S)

        for g in range(self.G):
            v_g = self.V[:, g]

            model = SGL(l1_ratio=self.lambda1, alpha=self.alpha, groups=groups)
            model.fit(S_scaled, v_g)
            self.C[:, g] = model.coef_

    def normalize_synergy(self, index: int) -> None:
        """Normalizes a synergy to remove scaling ambiguity.

        Params:
            index (int): Index of the synergy in s_list.
        Returns:
            None.
        """
        norm = np.linalg.norm(self.s_list[index].flatten())
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
            None.
        Returns:
            npt.NDArray: The V estimation, where the columns are grasping tasks, shape (nT, G).
        """
        return S @ C

    def C_loss(self) -> float:
        return np.sum((self.true_C - self.C)**2)
    
    def S_loss(self) -> float:
        return np.sum((self.true_S - self.S)**2)
    
    def V_loss(self) -> float:
        """Uses squared L2 norm to calculate V loss.

        Params:
            None.
        Returns:
            float: The loss (difference in actual and predicted value of V and V_est, respectively, squared).
        """
        return np.sum((self.V - self.V_est(self.S, self.C))**2)

    def compare_V(self, compare: npt.NDArray, epoch: int) -> None:
        """Comparison of V vs. estimated V.

        Params:
            compare (npt.NDArray): Numpy array to compare with V.
            epoch (int): Current epoch for alternating minimization.
        Returns:
            None.
        """
        V_stacked = self.V.flatten()
        V_est_stacked = self.V_est(self.S, self.C).flatten()

        plt.figure(figsize=(10,3))
        plt.plot(V_stacked, label="True v")
        plt.plot(V_est_stacked, label="Estimated v")
        plt.title(f"v reconstruction at epoch {epoch}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def active_synergies(self, tol: float = 1e-8) -> tuple[list[int], list[int], npt.NDArray]:
        active, dropped = [], []
        group_norms = np.zeros(self.m)
        for j in range(self.m):
            block = self.C[self.C_mask(j), :]
            group_norm = np.linalg.norm(block, ord='fro')
            group_norms[j] = group_norm
            (active if group_norm > tol else dropped).append(j)
        return active, dropped, group_norms

    @abstractmethod
    def solve(self) -> None:
        pass