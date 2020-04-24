import lib_non_local_games as nlg
import numpy as np
import cvxpy as cp
import os
import functools as fc
from operator import mul

# Before we start, check that MOSEK is installed
if not 'MOSEK' in cp.installed_solvers():
    raise RuntimeError('Please install MOSEK before running this main file.')

dimA1 = 2
dimA2 = 2
dimQ1 = 3
dimQ2 = 3

dimT = 2

# Subsystems A1 Q1 A2 Q2
subs_A1Q1A2Q2 = (dimA1,dimQ1,dimA2,dimQ2)
indices_A1Q1A2Q2 = nlg.indices_list(subs_A1Q1A2Q2)
dim_A1Q1A2Q2 = fc.reduce(mul, subs_A1Q1A2Q2, 1)

# I3322 rule for the inequality
def I3322_rule_ineq(a,x,b,y):
    
    value = str(a)+str(b)+str(x)+str(y) # Carefull with the order of this...
    
    factors_abxy = {'0010':1./3.,
                    '0020':1./3.,
                    '0001':1./3.,
                    '0011':2./3.,
                    '0021':-4./3.,
                    '0002':2./3.,
                    '0012':-1,
                    '0022':1,
                    '1000':-2./3.,
                    '1010':-2./3.,
                    '1020':-2./3.,
                    '0100':-1./3.,
                    '0101':-1./3.,
                    '0102':-1./3.,
                    '1001':-1./3.,
                    '1011':-1./3.,
                    '1021':-1./3.
                   }
    
    if value in factors_abxy.keys():
        return factors_abxy[value]
    else:
        return None

# Partial trace numpy
def partial_trace_numpy(rho, dims, axis=0):
    
    dims_ = np.array(dims)
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, dims_.size+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)

    return traced_out_rho.reshape([rho_dim, rho_dim])

# I operator, built out of the M_ax and N_by operators and the factors of the inequality under consideration
def I_operator(factors_ineq,indices_A1Q1A2Q2,dimQ1,dimQ2,dimT):
    
    qubit_tot = 2+dimQ1+dimQ2
    subs_tot = tuple(np.full(qubit_tot, dimT))
    dim_tot = fc.reduce(mul, subs_tot, 1)

    # Initial order for permutations
    init_order = np.arange(qubit_tot)

    # Swap operator between two subsystems
    F_12 = nlg.permutation_matrix((0,1), (1,0), (dimT, dimT))

    # First eigenstate of ZZ operator
    eigenstate_ZZ = np.zeros(dimT**2)
    eigenstate_ZZ[0] = 1
    eigenproj_ZZ = np.outer(eigenstate_ZZ,eigenstate_ZZ)

    # Identity operator (we use this one a few times in the for loop)
    Id = np.identity( dimT**(qubit_tot-2) )
    
    # Building the I operator
    I_list = []
    for a,x,b,y in indices_A1Q1A2Q2:
        
        factor = factors_ineq(a,x,b,y)
        
        if factor == None:
            continue

        M1 = a * np.identity(dimT**2) + (-1)**a * F_12
        N1 = b * np.identity(dimT**2) + (-1)**b * F_12

        M2 = nlg.tensor( [M1, Id] )
        N2 = nlg.tensor( [N1, Id] )

        # We need to reorder operator M2
        #
        # E.g. M2 = F_A|A1 x I_A0,B,A2,B0,B1,B2
        #
        # A  A1 A0 B  A2 B0 B1 B2
        # 0  1  2  3  4  5  6  7
        # A  B  A0 A1 A2 B0 B1 B2
        # 0  3  2  1  4  5  6  7
        #
        final_order = np.arange(qubit_tot)
        # Move Ax <-> B
        final_order[2+x] = 1
        final_order[1] = 2+x

        # Get the correct order for M
        P = nlg.permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
        M = P @ M2 @ P.T # Notice that M is Identity over B B1 B2 B3 (also on all A* apart from Ax)

        # We need to reorder operator N2
        #
        # E.g. M2 = F_B|B0 x I_A0,A1,A2,A,B1,B2
        #
        # B  B0 A0 A1 A2 A  B1 B2
        # 0  1  2  3  4  5  6  7
        # A  B  A0 A1 A2 B0 B1 B2
        # 5  0  2  3  4  1  6  7
        #
        final_order = np.arange(qubit_tot)
        # Move Bx -> A and B <-> A
        final_order[2+dimQ1+y] = 1
        final_order[0] = 2+dimQ1+y
        final_order[1] = 0

        # Get the correct order for N
        P = nlg.permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
        N = P @ N2 @ P.T # Notice that N it is Identity over A A1 A2 A3 (also on all B* apart from By)

        # Use assumption on W state to reduce M and N operator
        known_W = nlg.tensor( [eigenproj_ZZ,Id] )

        # Permute the operator M.N (it is in std order at the moment)
        #
        # E.g. get (A2 B2) A B A0 A1 B0 B1
        #
        # A  B  A0 A1 A2 B0 B1 B2
        # 0  1  2  3  4  5  6  7
        # A2 B2 A  B  A0 A1 B0 B1
        # 4  7  0  1  2  3  5  6
        #
        first_mask = np.array([dimQ1+1,dimQ1+dimQ2+1])
        second_mask = np.arange(dimQ1+1)
        third_mask = np.arange(dimQ1+2,dimQ1+dimQ2+1)
        final_order = np.concatenate((first_mask,second_mask,third_mask))

        # Build the overall operator
        P = nlg.permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
        Q = known_W @ P @ M @ N @ P.T

        # Reduce the dimension by tracing out the part of the operator we already know (last two subsystems)
        partial_dimension = [dimT**2,dimT**(qubit_tot-2)]
        I_list.append( factor * partial_trace_numpy(Q, partial_dimension) )
        
    I_op = np.sum(I_list,axis=0)
    
    return np.around(I_op)

## Variable
subs_W = tuple(np.full(dimQ1+dimQ2, dimT))
dim_W = fc.reduce(mul, subs_W, 1)

W = cp.Variable((dim_W,dim_W),hermitian=True)

## Obj function
I_operator = I_operator(I3322_rule_ineq,indices_A1Q1A2Q2,dimQ1,dimQ2,dimT)
objective_function = cp.trace(cp.matmul(W,I_operator))
    
## CONSTRAINTS

constraints = []
    
# 1) rho_TTSS are (sub-normalized) quantum states
# 1a) trace of the sum is 1
constraints.append( cp.trace(W) - 1 == 0 )

# 1b) positive semidefinite matrices
constraints.append( W >> 0 )

# 2) PPT all over
PPT_dim = (2,)*(dimQ1+dimQ2-1)
PPT_list = [np.concatenate((np.full(2,item[0]),item[1:])) for item in nlg.indices_list(PPT_dim)]

for PPT in PPT_list:
    
    if (sum(PPT) == 0) or (sum(PPT) == 6):
        continue
    
    constraints.append( nlg.partial_transpose(W,subs_W,tuple(PPT)) >> 0 )

# Write the problem
prob = cp.Problem(cp.Maximize(cp.real(objective_function)), constraints)

# Solve the problem
optimal_value = prob.solve(verbose=True,solver='MOSEK')

# Print the optimal value
print(optimal_value)
