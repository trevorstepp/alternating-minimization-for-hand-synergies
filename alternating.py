from sklearn import linear_model
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple
import seaborn as sns
import os

np.set_printoptions(precision=4, suppress=True)

class AlternatingMinimization:
	def __init__(self, T: int, t_s: int, m: int, v_list: list[npt.NDArray], num_joints: int=10, alpha: float=0.01, num_nonzero: Optional[int]=None, seed: Optional[int]=None) -> None:
		self.T = T  # duration of grasp
		self.t_s = t_s  # duration of synergy
		self.alpha = alpha  # for Lasso
		self.seed = seed
		self.max_shift = self.T - self.t_s + 1
		self.m = m  # number of synergies
		self.v_list = v_list # list of vectors containing velocities for grasping tasks
		self.num_joints = num_joints
		self.num_nonzero = num_nonzero or (self.m * self.max_shift) // 4  # number of nonzero entries in c_true
		#self.num_grasps = len(v_list)
		self.num_grasps = 2

		# **** Testing variables ****
		self.s_list_true = self.init_synergies() 
		self.S_true = self.build_S(self.s_list_true)
		self.c_true = self.sparse_c()
		self.v = self.get_v()
		# **** End ****

		#self.v = np.concatenate([v.flatten() for v in self.v_list])  # velocity vector of all grasping tasks stacked
		self.s_list: list[npt.NDArray] = self.init_synergies() # list of synergies
		self.S: Optional[npt.NDArray] = self.build_S(self.s_list)
		self.c: Optional[npt.NDArray] = self.sparse_c()
		#self.c: Optional[npt.NDArray] = None
		"""
		self.s_list: list[npt.NDArray] = self.s_list_true.copy() # list of synergies
		self.S: Optional[npt.NDArray] = self.build_S(self.s_list)
		"""
		#self.c: Optional[npt.NDArray] = self.c_true.copy()

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
	def build_S(self, s_list: npt.NDArray) -> npt.NDArray:
		# will build S column by column
		S_cols = []
		for j in range(self.m):
			s_j = s_list[j] # shape (num_joints, t_s)
			for shift in range(self.max_shift):
				new_col = self.s_shift(s_j, shift) # shape (num_joints * T,)
				S_cols.append(new_col)
		# turn into matrix (currently, S_cols is a list of 1-d vectors)
		S_single = np.column_stack(S_cols)  # shape (num_joints * T, m * max_shifts)
		# now add all grasps
		S = np.vstack([S_single for _ in range(self.num_grasps)])
		return S
	
	"""
	Rebuilds matrix S after s_j is updated

	Parameters:
		None
	Returns:
		None
	"""
	def update_S(self, s_list: npt.NDArray) -> None:
		self.S = self.build_S(s_list)

	"""
	Solves for c using Lasso regression

	Parameters:
		None
	Returns:
		A 1-d numpy array that is the new prediction for c
	"""
	def solve_c(self) -> None:
		# Lasso regression
		lasso = linear_model.Lasso(self.alpha, max_iter=5000, fit_intercept=False)
		lasso.fit(self.S, self.v)
		# get c estimate
		self.c = lasso.coef_
	
	def solve_S(self) -> npt.NDArray:
		#v_hat = self.S @ self.c
		for i in range(self.m):
			v_hat = self.S @ self.c
			# isolate the current synergy when calculating r_j
			# v - (v_hat - current synergy contributions)
			r_j = self.v - v_hat
			
			# we will skip certain rows of r_j and A_j if there are no contributions
			A_j_rows = [] 
			r_j_used = []

			# need to cover all instances of the current synergy
			for shift in range(self.max_shift):
				c_index = i * self.max_shift + shift # index of coefficient for j-th shift of i-th synergy
				coeff = self.c[c_index]
				if coeff == 0:
					continue

				# add contribution back
				r_j += coeff * self.S[:, c_index]

				for g in range(self.num_grasps):
					for joint in range(self.num_joints):
						for t in range(self.t_s):
							time_index = shift + t
							"""
							if time_index >= self.T:
								continue
							"""
							row = np.zeros(self.num_joints * self.t_s)
							local_index = joint * self.t_s + t
							row[local_index] = coeff

							global_index = g * self.num_joints * self.T + joint * self.T + time_index
							A_j_rows.append(row)
							r_j_used.append(r_j[global_index])
				
			if not A_j_rows:
				continue

			A_j = np.vstack(A_j_rows) # shape (num_rows, num_joints * t_s)
			r_j_new = np.array(r_j_used) # shape (num_rows,)

			s_flat, *_ = np.linalg.lstsq(A_j, r_j_new, rcond=None)

			# helps with problem of exploding S
			max_norm = 4.0
			norm = np.linalg.norm(s_flat)
			if norm > max_norm:
				s_flat = (s_flat / norm) * max_norm

			self.s_list[i] = np.reshape(s_flat, (self.num_joints, self.t_s))

			# must update S now for the next synergy's optimization
			self.update_S(self.s_list)

		return self.S

	"""
	Attempts to find S and c that solve v using alternating minimization

	Parameters:
		epochs (int): number of times alternating minimization should run
					  one run includes solving for c and solving for S
	Returns:
		None
	"""
	def alternatingMin(self, epochs: int=100) -> None:
		for epoch in range(epochs):
			self.solve_c()
			self.solve_S()
			print(f"Epoch {epoch}")
			print(f"S MAE: {self.S_loss()}")
			print(f"c loss: {self.c_loss()}")
			print(f"v loss {self.v_loss()}")
			print("-" * 30)
			if epoch % 10 == 0:
				self.compare_v(epoch)
	
	def c_loss(self) -> float:
		return np.sum((self.c_true - self.c)**2)
	
	def S_loss(self) -> float:
		return np.mean(np.abs(self.S_true - self.S))

	def v_loss(self) -> float:
		v_est = self.S @ self.c
		return np.sum((self.v - v_est)**2)

# **** FUNCTIONS BELOW ARE FOR TESTING PURPOSES ****
	# In an actual use case, the user chooses v
	# For testing, we generate S and c and do v = S @ c

	def sparse_c(self):
		if self.seed is not None:
			np.random.seed(self.seed)
		c_len = self.m * self.max_shift  # length of c is the number of columns of S
		c = np.zeros(c_len)
		nonzero_indices = np.random.choice(c_len, size=self.num_nonzero, replace=False)
		c[nonzero_indices] = np.random.randn(self.num_nonzero)
		return c

	def get_v(self) -> npt.NDArray:
		v = self.S_true @ self.c_true
		return v
	
	def compare_v(self, epoch: int):
		plt.figure(figsize=(10,3))
		plt.plot(self.v, label="True v")
		plt.plot(self.S @ self.c, label="Estimated v")
		plt.title(f"v reconstruction at epoch {epoch}")
		plt.xlabel("Time")
		plt.ylabel("Amplitude")
		plt.legend()
		plt.grid(True)
		plt.show()
	
	def compare_c(self):
		plt.figure()
		plt.stem(self.c_true, markerfmt='go', label="True c")
		plt.stem(self.c, markerfmt='rx', linefmt='r-', label="Estimated c")
		plt.legend()
		plt.title("c comparison")
		plt.xlabel("Index")
		plt.ylabel("Coefficient value")
		plt.show()

	def compare_synergies(self):
		for i, (s_true, s_est) in enumerate(zip(self.s_list_true, self.s_list)):
			plt.figure(figsize=(12, 4))
			for j in range(self.num_joints):
				plt.subplot(2, 5, j + 1)
				plt.plot(s_true[j], label="True", color='green')
				plt.plot(s_est[j], label="Estimated", color='red', linestyle='--')
				plt.title(f"Joint {j + 1}")
				plt.xticks([])
				plt.yticks([])
				if j == 0:
					plt.legend()
			plt.suptitle(f"Synergy {i + 1} - Joint-wise Comparison")
			plt.tight_layout()
			plt.show()

if __name__ == '__main__':
	T = 6
	t_s = 3
	m = 2
	num_joints = 10
	v_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
	test = AlternatingMinimization(T, t_s, m, v_list, num_joints)
	test.alternatingMin(epochs=100)
	"""
	for _ in range(5):
		#test.solve_S()
		test.solve_c()
		print("****")
		print("c loss:", test.c_loss())
		print("S loss:", test.S_loss())
		print(f"v loss: {test.v_loss()}")
	"""
	"""
	test.solve_c()
	c1 = test.c.copy()
	test.solve_S()
	test.solve_c()
	print(np.sum((c1 - test.c)**2))
	
	print(f"S_true = {test.S_true}")
	print("****")
	print(f"S = {test.S}")
	"""
	print(f"v length {len(test.v)}")
	print(f"c_true: {test.c_true}")
	print(f"c: {test.c}")
	test.compare_c()
	test.compare_synergies()