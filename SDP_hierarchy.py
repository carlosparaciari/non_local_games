import numpy as np
import cvxpy as cp
import functools as fc
from operator import mul

import lib_non_local_games as nlg

class SDP_relaxation:

	def __init__(self, level, answers, questions, dim_assistance, rule_function, distributions):

		# Level of the SDP relaxation
		self.n1, self.n2 = level

		# Dimension of questions and answers for the game
		self.dimA1, self.dimA2 = answers
		self.dimQ1, self.dimQ2 = questions

		# Dimension of the assisting quantum system
		self.dimT = dim_assistance
		self.dimS = dim_assistance # Copy due to swap-trick

		# Rules of the game
		self.rule = rule_function

		# Probability distribution of questions asked by referee
		self.probQ1, self.probQ2 = distributions

		# Useful variables for the methods of the class
		self.subs_A1Q1 = (self.dimA1,self.dimQ1)
		self.indices_A1Q1 = nlg.indices_list(self.subs_A1Q1)

		self.subs_A2Q2 = (self.dimA2,self.dimQ2)
		self.indices_A2Q2 = nlg.indices_list(self.subs_A2Q2)

		self.subs_A1Q1A2Q2 = self.subs_A1Q1 + self.subs_A2Q2
		self.indices_A1Q1A2Q2 = nlg.indices_list(self.subs_A1Q1A2Q2)

		self.subs_A1Q1A2Q2_ext = self.subs_A1Q1*self.n1 + self.subs_A2Q2*self.n2
		self.indices_A1Q1A2Q2_ext = nlg.indices_list(self.subs_A1Q1A2Q2_ext)

		self.subs_but_A1Q1A2Q2 = self.subs_A1Q1*(self.n1-1) + self.subs_A2Q2*(self.n2-1)
		self.indices_but_A1Q1A2Q2 = nlg.indices_list(self.subs_but_A1Q1A2Q2)

		self.subs_but_A1Q1 = self.subs_A1Q1*(self.n1-1) + self.subs_A2Q2*self.n2
		self.indices_but_A1Q1 = nlg.indices_list(self.subs_but_A1Q1)

		self.subs_but_A2Q2 = self.subs_A1Q1*self.n1 + self.subs_A2Q2*(self.n2-1)
		self.indices_but_A2Q2 = nlg.indices_list(self.subs_but_A2Q2)

		self.subs_TTSS = (self.dimT,self.dimT,self.dimS,self.dimS)

		self.subs_TTSS_ext = (self.dimT,)*(self.n1+self.n2) + (self.dimS,self.dimS)
		self.dim_TTSS_ext = fc.reduce(mul, self.subs_TTSS_ext, 1)

		# Maximally-entangled states between TT|SS
		self.F_TTSS = cp.Constant(nlg.permutation_matrix((0,1,2,3), (2,3,0,1), self.subs_TTSS))
		self.Phi_TTSS = nlg.partial_transpose(self.F_TTSS, self.subs_TTSS, (0,0,1,1))

		# Maximally-mixed state on T
		self.rhoT = np.identity(self.dimT)/self.dimT

		# Function mapping a sequence of indices to integers
		self.StI = lambda seq : nlg.seqtoint(seq, self.subs_A1Q1A2Q2_ext)

		# List of SDP variable
		self.rho_variable = []

		# Objective function of the problem
		self.object_function = 0

		# The list of constraints
		self.constraints = []

	# This method fills the list of variables with CVX 2D matrices of the correct size
	#
	# INPUT:
	#       - complex_field: boolean specifying whether the variable is hermitian or symmetric
	#
	def initialize_variable(self, complex_field=False):

		variable_shape = (self.dim_TTSS_ext,self.dim_TTSS_ext)

		for i in map(self.StI, self.indices_A1Q1A2Q2_ext):
			if complex_field:
				self.rho_variable.append( cp.Variable(variable_shape, hermitian=True) )
			else:
				self.rho_variable.append( cp.Variable(variable_shape, symmetric=True) )

	# This method creates the objective function for the game
	#
	def create_objective_function(self):

		self.object_function = (self.dimT**2) * sum([self.rule(*index) * cp.trace( self.Phi_TTSS @ self.rho_TTSS(*index) )
													 for index in self.indices_A1Q1A2Q2])

	# This method writes the SDP and returns the cvx problem
	#
	# INPUT:
	#       - complex_field: boolean specifying whether the variable is hermitian or symmetric
	#
	def write_problem(self, complex_field=False):
		if complex_field:
			problem = cp.Problem(cp.Maximize(cp.real(self.object_function)), self.constraints)
		else:
			problem = cp.Problem(cp.Maximize(self.object_function), self.constraints)

		return problem

	# This method imposes that the variable is a classical-quantum state
	#
	def state_constraint(self):

		# a) The trace of the sum of variables is 1
		self.constraints.append( sum([cp.trace(self.rho_variable[i]) for i in map(self.StI, self.indices_A1Q1A2Q2_ext)]) - 1 == 0 )

		# b) Each variable is a positive semidefinite matrix
		for i in map(self.StI, self.indices_A1Q1A2Q2_ext):
			self.constraints.append( self.rho_variable[i] >> 0 )

	# This method implements the permutation-invariance constraints for Alice and Bob
	#
	def full_permutation_constraints(self):

		# Order for Alice and Bob subsystems
		in_order_Alice = np.arange(self.n1)
		in_order_Bob = np.arange(self.n2)

		## Permutations on Alice side

		# Order for Bob subsystems (A2Q2T)_1 ... (A2Q2T)_n2 stays unchanged
		fin_order_Bob = np.copy(in_order_Bob)

		# All generators of symmetric group S_n1
		for i in range(self.n1-1):
			fin_order_Alice = np.copy(in_order_Alice)
			fin_order_Alice[i], fin_order_Alice[i+1] = in_order_Alice[i+1], in_order_Alice[i]
			self._permutation_constraint(fin_order_Alice,fin_order_Bob)

		## Permutations on Bob side

		# Order for Alice subsystems (A1Q1T)_1 ... (A1Q1T)_n1 stays unchanged
		fin_order_Alice = np.copy(in_order_Alice)

		# All generators of symmetric group S_n2
		for j in range(self.n2-1):
			fin_order_Bob = np.copy(in_order_Bob)
			fin_order_Bob[j], fin_order_Bob[j+1] = in_order_Bob[j+1], in_order_Bob[j]
			self._permutation_constraint(fin_order_Alice,fin_order_Bob)

	# This method implements the linear constraint for Alice side
	#
	def linear_constraint_Alice(self):

		# Create dimension tuple for the partial tracing
		sub_dim = (self.dimT, self.dimT**(self.n1+self.n2-1)*self.dimS**2)

		for q1 in range(self.dimQ1):
			for index_else in self.indices_but_A1Q1:
				indices_A1q1a2q2_ext = [np.append(np.array([a1,q1]),index_else) for a1 in range(self.dimA1)]
				indices_A1Q1a2q2_ext = [np.append(index_A1Q1,index_else) for index_A1Q1 in self.indices_A1Q1]

				lhs = sum([self.rho_variable[self.StI(index)] for index in indices_A1q1a2q2_ext])

				rhs_variable = sum([self.rho_variable[self.StI(index)] for index in indices_A1Q1a2q2_ext])
				rhs_partial = nlg.partial_trace(rhs_variable, sub_dim)
				rhs = self.probQ1[q1] * cp.kron(self.rhoT, rhs_partial)

				self.constraints.append( lhs - rhs == 0 )

	# This method implements the linear constraint for Bob side
	#
	def linear_constraint_Bob(self):

		# Permutation matrix (T1...Tn1)(T1...Tn2)(SS) -> (Tn2...T1)(T1...Tn1)(SS)
		order = np.arange(self.n1+self.n2+2)

		maskA = order[:self.n1]
		maskB = np.flip(order[self.n1:self.n1+self.n2])
		maskS = order[self.n1+self.n2:]
		mask = np.concatenate((maskB,maskA,maskS))

		P = cp.Constant(nlg.permutation_matrix(order, mask, self.subs_TTSS_ext))

		# Create dimension tuple for the partial tracing
		sub_dim = (self.dimT, self.dimT**(self.n1+self.n2-1)*self.dimS**2)

		for q2 in range(self.dimQ2):
			for index_else in self.indices_but_A2Q2:
				indices_a1q1A2q2_ext = [np.append(index_else,np.array([a2,q2])) for a2 in range(self.dimA2)]
				indices_a1q1A2Q2_ext = [np.append(index_else,index_A2Q2) for index_A2Q2 in self.indices_A2Q2]

				lhs_variable = sum([self.rho_variable[self.StI(index)] for index in indices_a1q1A2q2_ext])
				lhs = P @ lhs_variable @ P.T

				rhs_variable = sum([self.rho_variable[self.StI(index)] for index in indices_a1q1A2Q2_ext])
				rhs_permuted = P @ rhs_variable @ P.T
				rhs_partial = nlg.partial_trace(rhs_permuted, sub_dim)
				rhs = self.probQ2[q2] * cp.kron(self.rhoT,rhs_partial)

				self.constraints.append( lhs - rhs == 0 )

	# This method creates PPT constraints along all the cuts T_1 | ... | T_n1 | T_1 | ...| T_n2 | SS
	#
	def PPT_constraints(self):

		# Create tuple of choices (0=no-PT and 1=PT), one for each subsystem (SS subsystem counts as one)
		PT_choice = (2,)*(self.n1+self.n2+1)

		# Create all possible combinations of PT and no-PT allowed by the cuts
		PT_list = np.array([np.concatenate((item[:-1],np.full(2,item[-1]))) for item in nlg.indices_list(PT_choice)])

		# Remove trivial cases (all 0's and all 1's)
		num_subs = self.n1+self.n2+2
		bool_trivial = np.sum(PT_list,axis=1) % num_subs != 0
		PT_list = PT_list[bool_trivial]

		# Remove double cases and add PPT constraints
		used_PT_list = []

		for PT in PT_list:
			opposite_PT = (PT + 1) % 2
			is_PT_in = np.any([np.all(item == PT) or np.all(item == opposite_PT) for item in used_PT_list])
			if not is_PT_in:
				used_PT_list.append(PT) # We use this to decide whether to add new constraints or not
				for i in map(self.StI, self.indices_A1Q1A2Q2_ext):
					PPT_constr = nlg.partial_transpose(self.rho_variable[i],self.subs_TTSS_ext,PT) >> 0
					self.constraints.append(PPT_constr)

	# This method constructs the first level NPA constraint in terms of the optimsation variable (rho_variable)
	# NPA style constraint (see PhysRevLett.98.010401)
	#
	# INPUT:
	#       - proj: boolean variable on assuming projective measurements
	#
	def NPA1_constraint(self,proj=True):

		# Introduce the normalization factor
		renorm = lambda x,y : self.dimT**2/(self.probQ1[x]*self.probQ2[y])

		# The P matrix containing the information about the variables rho_variable is give by
		P = []

		for a1,q1 in self.indices_A1Q1:
			P_row = [renorm(q1,q2)*cp.trace(self.Phi_TTSS@self.rho_TTSS(a1,q1,a2,q2)) for a2,q2 in self.indices_A2Q2]
			P.append(P_row)

		P = cp.bmat(P)

		# The Q matrix containing the information about the variables rho_variable and also some new variables
		Q = []

		for a1,q1 in self.indices_A1Q1:
			Q_row = []
			for a1p,q1p in self.indices_A1Q1:
				if q1 == q1p:
					if a1 == a1p:
						val = sum([renorm(q1,q2)*cp.trace(self.Phi_TTSS@self.rho_TTSS(a1,q1,a2,q2)) for a2,q2 in self.indices_A2Q2])/self.dimQ2
					else:
						if proj:
							val = cp.Constant(0) # Assume projective measurements.
						else:
							val = cp.Variable() # Otherwise, just new variable
				else:
					val = cp.Variable()
				Q_row.append(val)
			Q.append(Q_row)

		Q = cp.bmat(Q)

		# The R matrix containing the information about the variables rho_variable and also some new variables
		R = []

		for a2,q2 in self.indices_A2Q2:
			R_row = []
			for a2p,q2p in self.indices_A2Q2:
				if q2 == q2p:
					if a2 == a2p:
						val = sum([renorm(q1,q2)*cp.trace(self.Phi_TTSS@self.rho_TTSS(a1,q1,a2,q2)) for a1,q1 in self.indices_A1Q1])/self.dimQ1
					else:
						if proj:
							val = cp.Constant(0) # Assume projective measurements.
						else:
							val = cp.Variable() # Otherwise, just new variable
				else:
					val = cp.Variable()
				R_row.append(val)
			R.append(R_row)

		R = cp.bmat(R)

		# Constructing vector v of dimension dimA1*dimQ1+dimA2*dimQ2
		v = []

		for a1,q1 in self.indices_A1Q1:
			v.append(sum([renorm(q1,q2)*cp.trace(self.Phi_TTSS@self.rho_TTSS(a1,q1,a2,q2)) for a2,q2 in self.indices_A2Q2])/self.dimQ2)

		for a2,q2 in self.indices_A2Q2:
			v.append(sum([renorm(q1,q2)*cp.trace(self.Phi_TTSS@self.rho_TTSS(a1,q1,a2,q2)) for a1,q1 in self.indices_A1Q1])/self.dimQ1)

		v = cp.bmat([v])
		w = cp.vstack([cp.Constant([[1]]),v.T])

		# Builiding the matrix M that should be positive semi-definite (NPA constraint)
		M = cp.vstack([cp.hstack([Q,P]),cp.hstack([P.T,R])])
		M = cp.vstack([v,M])
		M = cp.hstack([w,M])

		self.constraints.append( M >> 0 )
		self.constraints.append( M - M.T == 0 )

	# This method provides the reduced variable on the classical space A1Q1A2Q2 and on the quantum space TTSS
	#
	# INPUT:
	#       - a1,q1,a2,q2: the coordinate of the classical space A1Q1A2Q2
	#
	# OUTPUT:
	#       - rho_reduced: the reduced variable
	#
	def rho_TTSS(self,a1,q1,a2,q2):

		# Build the indices for the extended state
		index_a1q1 = np.array([[a1,q1]])
		index_a2q2 = np.array([[a2,q2]])

		indices_but_a2q2 = nlg.fuse_arrays(index_a1q1,self.indices_but_A1Q1A2Q2)
		indices_a1q1a2q2_ext = nlg.fuse_arrays(indices_but_a2q2,index_a2q2)

		# Reduce the classical part
		rho_reduced = sum([self.rho_variable[self.StI(index)] for index in indices_a1q1a2q2_ext])

		# Reduce the quanutm part
		if self.n1 != 1 or self.n2 != 1:
			dim_subsys = (self.dimT, self.dimT**(self.n1+self.n2-2), self.dimT*self.dimS**2)
			rho_reduced = nlg.partial_trace(rho_reduced, dim_subsys, axis=1)

		return rho_reduced

	# This private method implements a single constraint coming from the permutation-invariariance of the state
	#
	# INPUT:
	#       - order_Alice: the new order for Alice' subsystems
	#       - order_Bob: the new order for Bob' subsystems
	#
	def _permutation_constraint(self,order_Alice,order_Bob):

		# Order for the quantum systems
		init_order_QS = np.arange(self.n1 + self.n2 + 2)
		order_SS = np.arange(self.n1 + self.n2, self.n1 + self.n2 + 2)
		fin_order_qs = np.concatenate( (order_Alice, order_Bob + self.n1, order_SS) )

		# The permutation matrix swapping the quantum subsystems
		P = nlg.permutation_matrix(init_order_QS, fin_order_qs, self.subs_TTSS_ext)

		# The permutation function for the classical indices
		perm = lambda index : self._permute_index(index,order_Alice,order_Bob)

		for index in self.indices_A1Q1A2Q2_ext:
			lhs = self.rho_variable[self.StI(index)]

			rhs_variable = self.rho_variable[self.StI(perm(index))]
			rhs = P @ rhs_variable @ P.T

			self.constraints.append( lhs - rhs == 0 )


	# This private method permutes the indices according to a given permutation
	#
	# INPUT:
	#       - index: the index to be permuted
	#       - order_Alice: the new order for Alice's indices
	#       - order_Bob: the new order for Bob's indices
	#
	# OUTPUT:
	#       - permuted_index : the permuted array of indices
	#
	def _permute_index(self,index,order_Alice,order_Bob):

		permuted_index = np.empty(index.shape,dtype=index.dtype)

		for i,j in enumerate(order_Alice):
			permuted_index[2*i] = index[2*j] # permuting Alice answers (the a1's)
			permuted_index[2*i+1] = index[2*j+1] # permuting Alice questions(the q1's)

		for i,j in enumerate(order_Bob):
			permuted_index[2*self.n1+2*i] = index[2*self.n1+2*j] # permuting Bob answers (the a2's)
			permuted_index[2*self.n1+2*i+1] = index[2*self.n1+2*j+1] # permuting Bob questions(the q2's)

		return permuted_index
