from sklearn import linear_model
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple
import seaborn as sns
import os

np.set_printoptions(precision=4, suppress=True)

class LassoTest:
	# Note: This is currently for one grasping task (only requires slight modification to work for multiple)
	def __init__(self, T: int, t_s: int, m: int, num_joints: int=10, alpha: float=0.1, num_nonzero: Optional[int]=None, seed=None) -> None:
		self.T = T  # duration of grasp
		self.t_s = t_s  # duration of synergy
		self.alpha = alpha  # for Lasso
		#self.rng = np.random.default_rng(seed)
		self.seed = seed
		self.max_shift = self.T - self.t_s + 1
		self.m = m  # number of synergies
		self.num_joints = num_joints
		self.num_nonzero = num_nonzero or (self.m * self.max_shift) // 4  # number of nonzero entries in c_true

		self.s_list: Optional[list[npt.NDArray]] = None  # list of synergies
		self.S: Optional[npt.NDArray] = None 
		self.c_true: Optional[npt.NDArray] = None 
		self.v: Optional[npt.NDArray] = None  # v matrix formed from S @ c
		self.c_est: Optional[npt.NDArray] = None  # estimated c value from lasso using S and v

	"""
	single_run() should be used for standalone tests, run_once() should be used for large-scale tests
	(running multiple tests using a loop)

	Parameters:
		None
	Returns:
		Tuple that contains the count of nonzeros in c_true and c_est
	"""
	def single_run(self) -> Tuple[int, int]:
		# run lasso
		self.s_list: list[npt.NDArray] = self.init_synergies()
		self.S = self.build_S()
		self.c_true = self.sparse_c() # multiply with S to get v, then do lasso to try to get back
		self.v = self.get_v() # generate v
		self.c_est = self.solve_c(self.v) # recover c using Lasso

		# return number of nonzeros in each vector for comparison
		return np.count_nonzero(self.c_true), np.count_nonzero(self.c_est)
	
	def run_once(self, m: Optional[int]=None, num_joints: Optional[int]=None) -> Tuple[int, int]:
		# update variables related to synergies and c (if needed)
		if m is not None:
			self.m = m
			self.num_nonzero = (self.m * self.max_shift) // 4
		if num_joints is not None:
			self.num_joints = num_joints

		# run lasso
		self.s_list: list[npt.NDArray] = self.init_synergies()
		self.S = self.build_S()
		self.c_true = self.sparse_c() # multiply with S to get v, then do lasso to try to get back
		self.v = self.get_v() # generate v
		self.c_est = self.solve_c(self.v) # recover c using Lasso

		# return number of nonzeros in each vector for comparison
		return np.count_nonzero(self.c_true), np.count_nonzero(self.c_est)

	"""
	Initializes the synergies 

	Paramters:
		None
	Returns:
		A list of synergies, each of which is a 2-d numpy array
	"""
	def init_synergies(self) -> list[npt.NDArray]:
		if self.seed is not None:
			np.random.seed(self.seed)
		# return a list of m 2-d arrays (each of shape (num_joints, t_s))
		return [np.random.randn(self.num_joints, self.t_s) for _ in range(self.m)]
		#return [self.rng.standard_normal(size=(self.num_joints, self.t_s)) for _ in range(self.m)]

	"""
	Applies a time shift to each synergy and stretches to length T

	Parameters: 
		s (NDArray): 2-d array where each row is a joint and t_s cols
		shift (int): time shift for the synergy (every joint)
	Returns: 
		A 1-d numpy array that has been flattened/stacked
	"""
	def s_shift(self, s: npt.NDArray, shift: int) -> npt.NDArray:
		shifts = []  # to hold shifted rows of s
		for j in range(self.num_joints):
			front_pad = np.zeros(shift)
			back_pad = np.zeros(self.T - self.t_s - shift)
			shift_j = np.concatenate([front_pad, s[j], back_pad])
			shifts.append(shift_j)
		return np.concatenate(shifts)  # shape (num_joints * T,)

	# S = [ D_{11}s^1 | ... | D_{1K_1}s^1 | ... | D_{m1}s^m | ... | D_{mK_m}s^m ]
	def build_S(self) -> npt.NDArray:
		# will build S column by column
		S_cols = []
		for j in range(self.m):
			s_j = self.s_list[j] # shape (num_joints, t_s)
			for shift in range(self.max_shift):
				new_col = self.s_shift(s_j, shift) # shape (num_joints * T,)
				S_cols.append(new_col)
		# turn into matrix (currently, S_cols is a list of 1-d vectors)
		return np.column_stack(S_cols)  # shape (num_joints * T, m * max_shifts)
	
	def sparse_c(self):
		if self.seed is not None:
			np.random.seed(self.seed)
		c_len = self.S.shape[1]  # length of c is the number of columns of S
		c = np.zeros(c_len)
		nonzero_indices = np.random.choice(c_len, size=self.num_nonzero, replace=False)
		c[nonzero_indices] = np.random.randn(self.num_nonzero)
		return c

	def get_v(self) -> npt.NDArray:
		v = self.S @ self.c_true
		return v

	"""
	Solves for c using Lasso regression

	Parameters:
		None
	Returns:
		A 1-d numpy array that is the new prediction for c
	"""
	def solve_c(self, v: npt.NDArray) -> npt.NDArray:
		# Lasso regression
		lasso = linear_model.Lasso(self.alpha, fit_intercept=False)
		lasso.fit(self.S, v)
		# get c estimate
		c_est = lasso.coef_
		return c_est
	
	def c_loss(self) -> float:
		return np.sum((self.c_true - self.c_est)**2)

	def test_lasso(self, max_m: int=11, max_joint: int=11) -> Tuple[pd.DataFrame, pd.DataFrame]:
		data = []
		vectors = []
		temp = self.num_joints # used to reset num_joints each time inner loop finishes
		for i in range(self.m, max_m):
			for j in range(self.num_joints, max_joint):
				c_true_nonzeros, c_est_nonzeros = self.run_once(m=i, num_joints=j)
				loss = self.c_loss()

				data.append({
					'Synergies': i,
					'Repeats': self.max_shift,
					'Joints': j,
					'Loss': round(loss, 4)
				})

				df = pd.DataFrame({
					'(Synergies, Joints)': f"{(i, j)}",
					'c_true': np.round(self.c_true, 2),
					'c_est': np.round(self.c_est, 2),
					'Individual Loss': [round(x - y, 4) for x, y in zip(self.c_true, self.c_est)]
				})
				vectors.append(df)

			self.num_joints = temp

		df_data = pd.DataFrame(data)
		df_vectors = pd.concat(vectors, ignore_index=True)
		return df_data, df_vectors

def plot_data(df: pd.DataFrame) -> None:
	new_df = pd.pivot(data=df, index='Joints', columns='Synergies', values='Loss')
	heatmap = sns.heatmap(new_df, annot=True)
	heatmap.invert_yaxis()
	plt.tight_layout()
	name = "heatmap.png"
	plt.savefig(name)
	os.system(f"open {name}")

if __name__ == '__main__':
	T = 6
	t_s = 3
	m = 2
	num_joints = 1
	test = LassoTest(T, t_s, m, num_joints, alpha=0.01, seed=None)

	df_data, df_vectors = test.test_lasso()
	plot_data(df_data)

	"""
	for row in test.S:
		print("  ".join(f"{val:8.4f}" for val in row))
	"""
	#"""
	test.single_run()

	print("c_true:\n[", " ".join(f"{entry:.3f}" for entry in test.c_true), "]")
	print("VS")
	print("c_est:\n[", " ".join(f"{val:.3f}" for val in test.c_est), "]")
	print(f"\nLoss: {test.c_loss()}")
	print(f"V=SC Loss: {np.linalg.norm(test.v - test.S @ test.c_est)}")

	plt.stem(test.c_true, linefmt='g-', markerfmt='go', basefmt=' ')
	plt.stem(test.c_est, linefmt='r--', markerfmt='ro', basefmt=' ')
	plt.legend(['True c', 'Estimated c'])
	plt.xlabel('Entry in c')
	plt.ylabel('Value of entry')
	plt.show()
	#"""
	print(f"Size of S: {np.shape(test.S)}")
	cond = np.linalg.cond(test.S)
	print(f"Condition number of S: {cond:.4f}")