import lib_non_local_games as nlg
import numpy as np
import cvxpy as cp
import os
import functools as fc
from operator import mul

""" NOTE: This code will optimise over 4 times more variables, and each variable
		  will be 4 times bigger than the n1=2 n2=1 relaxation. As a result, we
		  should expect a demand for 16 times the RAM we used in the previous
		  case. This bring the RAM usage of this script to approx 100 GB.
	NOTE: Writing the constraints and phrasing the problem (cp.Problem) uses
		  approximatively 3 GB of RAM. Comparing with the previous relaxation,
		  which uses approx 0.3 GB, we get consistent prediction that the RAM
		  usage will increase by ~ 10 times.
	NOTE: Asking for the matrix variables to be symmetric (we consider real matrices
		  at the moment) will reduce by almost half the total number of parameters.
		  We find that this will slow down the first part of cp.Solve, where the
		  problem is recased to fit the specific solver we use (MOSEK), but will
		  substantially speed up the second part, where the problem is solved. Also,
		  solving the problem requires approximately half the RAM compared to before.
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

### CHSH game n_1 = 3 and n_2 = 1

# Subsystems A(1)1 A(2)1 A(3)1 Q(1)1 Q(2)1 Q(3)1 A2  Q2 
subs_AAAQQQ1_AQ2 = (dimA1,dimA1,dimA1,dimQ1,dimQ1,dimQ1,dimA2,dimQ2)
indices_AAAQQQ1_AQ2 = nlg.indices_list(subs_AAAQQQ1_AQ2)

# Subsystems A(1)1 Q(1)1 A(1)2 Q(1)2
subs_A1Q1A2Q2 = (dimA1,dimQ1,dimA2,dimQ2)
indices_A1Q1A2Q2 = nlg.indices_list(subs_A1Q1A2Q2)

# Subsystems T(1)1 T(2)1 T(3)1 T2 S1 S2
subs_TTT1_T2_SS = (dimT,dimT,dimT,dimT,dimS,dimS)
dim_TTT1_T2_SS = fc.reduce(mul, subs_TTT1_T2_SS, 1)

# Subsystems T(1)2 T(1)3 T2 S1 S2
subs_TTT_SS = (dimT,dimT,dimT,dimS,dimS)
dim_TTT_SS = fc.reduce(mul, subs_TTT_SS, 1)

# Subsystems A(2)1 Q(1)1 Q(2)1 A(1)2 A(2)2 Q(1)2 Q(2)2
subs_AAQQQ1_AQ2 = (dimA1,dimA1,dimQ1,dimQ1,dimQ1,dimA2,dimQ2)
indices_AAQQQ1_AQ2 = nlg.indices_list(subs_AAQQQ1_AQ2)

# Subsystems A1 Q1
subs_AQ1 = (dimA1,dimQ1)
indices_AQ1 = nlg.indices_list(subs_AQ1)

# Subsystems A(1)1 A(2)1 Q(1)1 Q(2)1 A(2)2 Q(1)2 Q(2)2
subs_AAQQ1_AQQ2 = (dimA1,dimA1,dimQ1,dimQ1,dimA2,dimQ2,dimQ2)
indices_AAQQ1_AQQ2 = nlg.indices_list(subs_AAQQ1_AQQ2)

# Subsystems A2 Q2
subs_AQ2 = (dimA2,dimQ2)
indices_AQ2 = nlg.indices_list(subs_AQ2)

# Subsystems A1 Q1 A2 Q2
subs_AQ1AQ2 = (dimA1,dimQ1,dimA2,dimQ2)
indices_AQ1AQ2 = nlg.indices_list(subs_AQ1AQ2)

# Subsystems A(1)1 A(1)2 Q(1)1 Q(2)1
subs_AAQQ1 = (dimA1,dimA1,dimQ1,dimQ1)
indices_AAQQ1 = nlg.indices_list(subs_AAQQ1)

# Subsystems T(1)* T(2)*
subs_TT = (dimT,dimT)
dim_TT = fc.reduce(mul, subs_TT, 1)

# Subsystems T(1)1 T(1)2 S1 S2
subs_TT_SS = (dimT,dimT,dimS,dimS)
dim_TT_SS = fc.reduce(mul, subs_TT_SS, 1)

# State on subsystem T
rhoT = np.identity(dimT)/dimT

## VARIABLES 

# The (sub-normalized) states we optimize over
rho_TTTTSS = []
shape_TTTTSS = (dim_TTT1_T2_SS, dim_TTT1_T2_SS)

for i in map(nlg.binarytoint,indices_AAAQQQ1_AQ2):
    rho_TTTTSS.append( cp.Variable(shape_TTTTSS,symmetric=True) )

## OBJECTIVE FUNCTION

# The swap operator takes T(1)1 T(2)1 T(3)1 T2 S1 S2 to S1 T(2)1 T(3)1 S2 T(1)1 T2
F_TTTTSS = nlg.permutation_matrix((0,1,2,3,4,5), (4,1,2,5,0,3), subs_TTT1_T2_SS)

# The object function is
obj_fun_components = []

for a1,q1,a2,q2 in indices_A1Q1A2Q2:
    v_a1q1a2q2 = int( nlg.CHSH_rule_function_A1Q1A2Q2(a1,q1,a2,q2) )
    variable_TTTTSS = sum([rho_TTTTSS[nlg.binarytoint([a1,a11,a12,q1,q11,q12,a2,q2])]
    					   for a11,a12,q11,q12 in indices_AAQQ1])
    obj_fun_components.append( v_a1q1a2q2 * cp.trace( cp.matmul(F_TTTTSS,variable_TTTTSS) ) )
    
object_function = cp.Constant(dimT**2) * sum(obj_fun_components)

## CONSTRAINTS

constraints = []

# 1) rho_TTTTSS are (sub-normalized) quantum states
# 1a) trace of the sum is 1
trace_rho = sum([cp.trace(rho_TTTTSS[i]) for i in map(nlg.binarytoint,indices_AAAQQQ1_AQ2)])
constraints.append( trace_rho - 1 == 0 )

# 1b) positive semidefinite matrices
for i in map(nlg.binarytoint,indices_AAAQQQ1_AQ2):
    constraints.append( rho_TTTTSS[i] >> 0 )

# 2) Permutation invariance
# 2a) over A(1)1 Q(1)1 T(1)1 and A(2)1 Q(2)1 T(2)1

# The permutation matrix swaps T(1)1 and T(2)1
P = nlg.permutation_matrix((0,1,2,3,4,5), (1,0,2,3,4,5), subs_TTT1_T2_SS)

for a1,a11,a12,q1,q11,q12,a2,q2 in indices_AAAQQQ1_AQ2:
    lhs = rho_TTTTSS[nlg.binarytoint([a1,a11,a12,q1,q11,q12,a2,q2])]
    
    rhs_variable = rho_TTTTSS[nlg.binarytoint([a11,a1,a12,q11,q1,q12,a2,q2])]
    rhs = cp.matmul(cp.matmul(P,rhs_variable),P)
    
    constraints.append( lhs - rhs == 0 )

# 2b) over A(1)1 Q(1)1 T(1)1 and A(3)1 Q(3)1 T(3)1

# The permutation matrix swaps T(1)1 and T(3)1
P = nlg.permutation_matrix((0,1,2,3,4,5), (2,1,0,3,4,5), subs_TTT1_T2_SS)

for a1,a11,a12,q1,q11,q12,a2,q2 in indices_AAAQQQ1_AQ2:
    lhs = rho_TTTTSS[nlg.binarytoint([a1,a11,a12,q1,q11,q12,a2,q2])]
    
    rhs_variable = rho_TTTTSS[nlg.binarytoint([a12,a11,a1,q12,q11,q1,a2,q2])]
    rhs = cp.matmul(cp.matmul(P,rhs_variable),P)
    
    constraints.append( lhs - rhs == 0 )

# 2c) over A(2)1 Q(2)1 T(2)1 and A(3)1 Q(3)1 T(3)1

# The permutation matrix swaps T(2)1 and T(3)1
P = nlg.permutation_matrix((0,1,2,3,4,5), (0,2,1,3,4,5), subs_TTT1_T2_SS)

for a1,a11,a12,q1,q11,q12,a2,q2 in indices_AAAQQQ1_AQ2:
    lhs = rho_TTTTSS[nlg.binarytoint([a1,a11,a12,q1,q11,q12,a2,q2])]
    
    rhs_variable = rho_TTTTSS[nlg.binarytoint([a1,a12,a11,q1,q12,q11,a2,q2])]
    rhs = cp.matmul(cp.matmul(P,rhs_variable),P)
    
    constraints.append( lhs - rhs == 0 )

# 3) First linear constraint
for a11,a12,q1,q11,q12,a2,q2 in indices_AAQQQ1_AQ2:
	indices_A1a11a12q1q11q12a2q2 = [nlg.binarytoint([a,a11,a12,q1,q11,q12,a2,q2])
									for a in range(dimA1)]
	indices_A1a11a12Q1q11q12a2q2 = [nlg.binarytoint([a,a11,a12,q,q11,q12,a2,q2])
									for a,q in indices_AQ1]

	lhs = sum([rho_TTTTSS[i] for i in indices_A1a11a12q1q11q12a2q2])

	rhs_variable = sum([rho_TTTTSS[i] for i in indices_A1a11a12Q1q11q12a2q2])
	rhs_partial = nlg.partial_trace(rhs_variable, [dimT, dim_TTT_SS])
	rhs = probQ1[q1] * cp.kron(rhoT, rhs_partial)
    
	constraints.append( lhs - rhs == 0 )

# 4) Second linear constraint

# The permutation matrix swaps T(1)1 and T2
P = cp.Constant(nlg.permutation_matrix((0,1,2,3,4,5), (3,1,2,0,4,5), subs_TTT1_T2_SS))

for a1,a11,a12,q1,q11,q12,q2 in indices_AAAQQQ1_Q2:
	indices_a1a11a12q1q11q12A2q2 = [nlg.binarytoint([a1,a11,a12,q1,q11,q12,a,q2])
									for a in range(dimA2)]
	indices_a1a11a12q1q11q12A2Q2 = [nlg.binarytoint([a1,a11,a12,q1,q11,q12,a,q])
									for a,q in indices_AQ2]

	lhs_variable = sum([rho_TTTTSS[i] for i in indices_a1a11a12q1q11q12A2q2])
	lhs = cp.matmul(cp.matmul(P,lhs_variable),P)

	rhs_variable = sum([rho_TTTTSS[i] for i in indices_a1a11a12q1q11q12A2Q2])
	rhs_permuted = cp.matmul(cp.matmul(P,rhs_variable),P)
	rhs_partial = nlg.partial_trace(rhs_permuted, [dimT, dim_TTT_SS])
	rhs = probQ2[q2] * cp.kron(rhoT, rhs_partial)

	constraints.append( lhs - rhs == 0 )

# 5) PPT criterium
#for i in map(nlg.binarytoint,indices_AAQQ1_AAQQ2):
#    constraints.append( nlg.partial_transpose(rho_TTTTSS[i],subs_TT1_TT2_SS,(0,0,0,0,1,1)) >> 0 )
#    constraints.append( nlg.partial_transpose(rho_TTTTSS[i],subs_TT1_TT2_SS,(1,1,0,0,0,0)) >> 0 )
#    constraints.append( nlg.partial_transpose(rho_TTTTSS[i],subs_TT1_TT2_SS,(0,0,1,1,0,0)) >> 0 )

## PROBLEM

# Write the problem
prob = cp.Problem(cp.Maximize(object_function), constraints)

# Solve the problem
optimal_value = prob.solve(verbose=True,solver='MOSEK')

# Print the optimal value
print(optimal_value)

## OPTIMAL RHO

# Create folder if it does not exist
os.makedirs("./optimal_rho", exist_ok=True)

# The permutation takes [T(1)1 T(2)1] [T(1)2 T(2)2] S1 S2 to [T(2)1 T(2)2] [T(1)1 T(1)2] S1 S2
#P = cp.Constant(nlg.permutation_matrix((0,1,2,3,4,5), (1,3,0,2,4,5), subs_TT1_TT2_SS))

for a1,q1,a2,q2 in indices_AQ1AQ2:
    indices_a1q1a2q2 = [nlg.binarytoint([a1,a11,a12,q1,q11,q12,a2,q2])
    					for a11,a12,q11,q12 in indices_AAQQ1]
    
    rho_variable = sum([rho_TTTTSS[i] for i in indices_a1q1a2q2])
    #rho_permuted = cp.matmul(cp.matmul(P,rho_variable),P)
    rho_TTSS = nlg.partial_trace(rho_variable, [dim_TT, dim_TT_SS])
    
    np.save('./optimal_rho/rho_a1_{}_q1_{}_a2_{}_q2_{}.npy'.format(a1,q1,a2,q2),
    	    np.array(rho_TTSS.value)
    	   )
