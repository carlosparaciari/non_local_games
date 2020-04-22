import lib_non_local_games as nlg
import numpy as np
import cvxpy as cp
import os
import functools as fc
from operator import mul

""" NOTE: Here we extend (A1Q1T) and (SS) twice.

		  The size of each variables is 128x128 (since 2^2 comes from A1Q1T^2, 2 from A2Q2T, and 2^4 from SS^2)
		  The number of variables is 64 (since 2^4 comes from A1Q1T^2, and 2^2 from A2Q2T)
		  Total number of scalar variables is 2^20

		  In comparison, for SDP31 we had a variable size of 64x64, and a number of variables of 256,
		  with a total number of scalar variables of 2^20.

		  It is likely that this program will run in a similar time to SDP31, requiring the same amount of memory

	NOTE: Mario suggested to use other solvers than Mosek, eg sdpt3, sedumi
"""

# Before we start, check that MOSEK is installed
if not 'MOSEK' in cp.installed_solvers():
    raise RuntimeError('Please install MOSEK before running this main file.')

# Parameter of the CHSH game
dimA1 = 2
dimA2 = 2
dimQ1 = 2
dimQ2 = 2

dimT = 2
dimS = 2

probQ1 = (.5,.5)
probQ2 = (.5,.5)

### CHSH game n_(A1Q1T) = 2 and n_(A2Q2T) = 1 and n_(SS) = 2

# Subsystems A(1)1 A(2)1 Q(1)1 Q(2)1 A2 Q2
subs_AAQQ1_AQ2 = (dimA1,dimA1,dimQ1,dimQ1,dimA2,dimQ2)
indices_AAQQ1_AQ2 = nlg.indices_list(subs_AAQQ1_AQ2)

# Function for mapping indices to integers
BtI = lambda seq : nlg.seqtoint(seq, subs_AAQQ1_AQ2)

# Subsystems T(1)1 T(2)1 T2 S(1)1 S(1)2 S(2)1 S(2)2
subs_TT1_T2_SSSS = (dimT,dimT,dimT,dimS,dimS,dimS,dimS)
dim_TT1_T2_SSSS = fc.reduce(mul, subs_TT1_T2_SSSS, 1)

# Subsystems A(1)1 Q(1)1 A2 Q2
subs_A1Q1A2Q2 = (dimA1,dimQ1,dimA2,dimQ2)
indices_A1Q1A2Q2 = nlg.indices_list(subs_A1Q1A2Q2)

# Subsystems A1 Q1
subs_A1Q1 = (dimA1,dimQ1)
indices_A1Q1 = nlg.indices_list(subs_A1Q1)

# Subsystems A2 Q2
subs_A2Q2 = (dimA2,dimQ2)
indices_A2Q2 = nlg.indices_list(subs_A2Q2)

# Subsystems T(1)1 T2 S(1)1 S(1)2
subs_TT_SS = (dimT,dimT,dimS,dimS)
dim_TT_SS = fc.reduce(mul, subs_TT_SS, 1)

# Subsystems T(2)1 S(2)1 S(2)2
subs_T_SS = (dimT,dimS,dimS)
dim_T_SS = fc.reduce(mul, subs_T_SS, 1)

# Subsystems A(2)1 Q(1)1 Q(2)1 A2 Q2
subs_AQQ1_AQ2 = (dimA1,dimQ1,dimQ1,dimA2,dimQ2)
indices_AQQ1_AQ2 = nlg.indices_list(subs_AQQ1_AQ2)

# Subsystems A(1)1 A(2)1 Q(1)1 Q(2)1 Q2
subs_AAQQ1_Q2 = (dimA1,dimA1,dimQ1,dimQ1,dimQ2)
indices_AAQQ1_Q2 = nlg.indices_list(subs_AAQQ1_Q2)

# Subsystems T(2)1 T2 S(1)1 S(1)2 S(2)1 S(2)2
subs_T1_T2_SSSS = (dimT,dimT,dimS,dimS,dimS,dimS)
dim_T1_T2_SSSS = fc.reduce(mul, subs_T1_T2_SSSS, 1)

# State on subsystem T
rhoT = np.identity(dimT)/dimT

## VARIABLES 

# The (sub-normalized) states we optimize over
rho = []
shape_rho = (dim_TT1_T2_SSSS, dim_TT1_T2_SSSS)

for i in map(BtI,indices_AAQQ1_AQ2):
    rho.append( cp.Variable(shape_rho,symmetric=True) ) # I am making them symmetric rather than hermitian

## OBJECTIVE FUNCTION

# Maximally entangled vectors between T|S and TT|SS
phi_TS = nlg.bipartite_unnorm_max_entangled_state(dimT)
phi_TSTS = nlg.tensor([phi_TS,phi_TS])

# Maximally mixed states between TT|SS (correct order of subsystems)
Phi_TSTS = np.outer(phi_TSTS,phi_TSTS)
P = nlg.permutation_matrix((0,1,2,3), (0,2,1,3), (dimT,dimT,dimT,dimT))
Phi_TTSS = P @ Phi_TSTS @ P

# The object function is
obj_fun_components = []

# Permute the state so that we have first T(1)1_T2_SS and then T(2)1_SS
P = nlg.permutation_matrix((0,1,2,3,4,5,6), (0,2,3,4,1,5,6), subs_TT1_T2_SSSS)

for a1,q1,a2,q2 in indices_A1Q1A2Q2:

	variable_TT1_T2_SS_SS = sum([rho[ BtI([a1,aa1,q1,qq1,a2,q2]) ] for aa1,qq1 in indices_A1Q1])
	variable_permuted = cp.matmul( cp.matmul(P,variable_TT1_T2_SS_SS) , P )
	variable_partial = nlg.partial_trace(variable_permuted, [dim_TT_SS,dim_T_SS], axis=1) # we trace out T(2)1_SS

	v_a1q1a2q2 = int( nlg.CHSH_rule_function_A1Q1A2Q2(a1,q1,a2,q2) )

	obj_fun_components.append( v_a1q1a2q2 * cp.trace( cp.matmul(Phi_TTSS,variable_partial) ) )

object_function = cp.Constant(dimT**2) * sum(obj_fun_components)

## CONSTRAINTS

constraints = []

# 1) rho variables are (sub-normalized) quantum states
# 1a) trace of the sum is 1
trace_rho = sum([cp.trace(rho[i]) for i in map(BtI,indices_AAQQ1_AQ2)])
constraints.append( trace_rho - 1 == 0 )

# 1b) positive semidefinite matrices
for i in map(BtI,indices_AAQQ1_AQ2):
    constraints.append( rho[i] >> 0 )

# 2) Permutation invariance
# 2a) over A(1)1 Q(1)1 T(1)1 and A(2)1 Q(2)1 T(2)1

# The permutation matrix swaps T(1)1 and T(2)1
P = nlg.permutation_matrix((0,1,2,3,4,5,6), (1,0,2,3,4,5,6), subs_TT1_T2_SSSS)

for a1,aa1,q1,qq1,a2,q2 in indices_AAQQ1_AQ2:
    lhs = rho[BtI([a1,aa1,q1,qq1,a2,q2])]
    
    rhs_variable = rho[BtI([aa1,a1,qq1,q1,a2,q2])]
    rhs = cp.matmul(cp.matmul(P,rhs_variable),P)
    
    constraints.append( lhs - rhs == 0 )

# 2b) over S(1)1 S(1)2 and S(2)1 S(2)2

# The permutation matrix swaps S(1)1 S(1)2 and S(2)1 S(2)2
P = nlg.permutation_matrix((0,1,2,3,4,5,6), (0,1,2,5,6,3,4), subs_TT1_T2_SSSS)

for i in map(BtI,indices_AAQQ1_AQ2):
    lhs = rho[i]
    rhs = cp.matmul(cp.matmul(P,rho[i]),P)
    
    constraints.append( lhs - rhs == 0 )

# 3) First linear constraint
for aa1,q1,qq1,a2,q2 in indices_AQQ1_AQ2:
	indices_A1aa1q1qq1a2q2 = [BtI([a,aa1,q1,qq1,a2,q2]) for a in range(dimA1)]
	indices_A1aa1Q1qq1a2q2 = [BtI([a,aa1,q,qq1,a2,q2]) for a,q in indices_A1Q1]

	lhs = sum([rho[i] for i in indices_A1aa1q1qq1a2q2])

	rhs_variable = sum([rho[i] for i in indices_A1aa1Q1qq1a2q2])
	rhs_partial = nlg.partial_trace(rhs_variable, [dimT, dim_T1_T2_SSSS])
	rhs = probQ1[q1] * cp.kron(rhoT, rhs_partial)
    
	constraints.append( lhs - rhs == 0 )

# # 4) Second linear constraint

# The permutation matrix swaps T(1)1 and T2
P = nlg.permutation_matrix((0,1,2,3,4,5,6), (2,1,0,3,4,5,6), subs_TT1_T2_SSSS)

for a1,aa1,q1,qq1,q2 in indices_AAQQ1_Q2:
	indices_a1aa1q1qq1A2q2 = [BtI([a1,aa1,q1,qq1,a,q2]) for a in range(dimA2)]
	indices_a1aa1q1qq1A2Q2 = [BtI([a1,aa1,q1,qq1,a,q]) for a,q in indices_A2Q2]

	lhs_variable = sum([rho[i] for i in indices_a1aa1q1qq1A2q2])
	lhs = cp.matmul(cp.matmul(P,lhs_variable),P)

	rhs_variable = sum([rho[i] for i in indices_a1aa1q1qq1A2Q2])
	rhs_permuted = cp.matmul(cp.matmul(P,rhs_variable),P)
	rhs_partial = nlg.partial_trace(rhs_permuted, [dimT, dim_T1_T2_SSSS])
	rhs = probQ2[q2] * cp.kron(rhoT, rhs_partial)

	constraints.append( lhs - rhs == 0 )

# 5) PPT criterium
for i in map(BtI,indices_AAQQ1_Q2):
    constraints.append( nlg.partial_transpose(rho[i],subs_TT1_T2_SSSS,(1,1,0,0,0,0,0)) >> 0 )
    constraints.append( nlg.partial_transpose(rho[i],subs_TT1_T2_SSSS,(0,0,1,0,0,0,0)) >> 0 )
    constraints.append( nlg.partial_transpose(rho[i],subs_TT1_T2_SSSS,(0,0,0,1,1,1,1)) >> 0 )

## PROBLEM

# Write the problem
prob = cp.Problem(cp.Maximize(object_function), constraints)

# Solve the problem
optimal_value = prob.solve(verbose=True,solver='MOSEK')

# Print the optimal value
print(optimal_value)