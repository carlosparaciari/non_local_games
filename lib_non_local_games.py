import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
import functools as fc
from operator import mul

# This function returns the computational basis of an Hilbert space with given dimension dim.
#
def basis(dim):
    basis = []
    for i in range(dim):
        vec = np.zeros(dim)
        vec[i] = 1
        basis.append(vec)
    return basis

# This function returns the tensor product of n states of different dimensions, passed as a list.
#
def tensor(array_list):
    return fc.reduce(lambda x,y : np.kron(x,y),array_list)

# This function fuses two arrays of arrays
#
# E.g. Given two vectors
#    
#     v = [[1,2],
#          [3,4],
#          [5,6]]
#    
#     w = [[7,8,9],
#          [10,11,12]]
#    
#     we get
#    
#     output = [[1,2,7,8,9],
#               [1,2,10,11,12],
#               [3,4,7,8,9],
#               [3,4,10,11,12],
#               [5,6,7,8,9],
#               [5,6,10,11,12]]
#
def fuse_arrays(v,w):
    
    if v.size == 0:
        return w
    
    if w.size == 0:
        return v
    
    fused_vw = np.empty((0,v.shape[1]+w.shape[1]))
    broadcast_shape = (w.shape[0],v.shape[1])

    for subv in v:
        broad_suv = np.broadcast_to(subv,broadcast_shape)
        fused_vw = np.vstack((fused_vw,np.hstack((broad_suv,w))))

    return fused_vw

# This function returns the unnormalized bipartite maximally entangled state of dimension dim.
#
def bipartite_unnorm_max_entangled_state(dim):
	return sum([tensor([v,v]) for v in basis(dim)])

# This function returs the list of indices for a tuple of subsystem of given dimension
#
# NOTE: the output list is **not** ordered.
#
def indices_list(dimension_tuple):
    number_subsystems = len(dimension_tuple)
    if number_subsystems != 0:
        subindices = [range(dim) for dim in dimension_tuple]
        return np.array(np.meshgrid(*subindices)).T.reshape(-1,number_subsystems)
    else:
        return np.array([])

# This function produces the permutation matrix for a given permutation of n subsystems of given dimensions
#
# Inputs:
#			- initial order: tuple of ordered number from 0 to n-1
#			- final_order: tuple of re-ordered numbers, describing the permutations
#			- dimension_subsystems: tuple of the dimension of each subsystem
#
def permutation_matrix(initial_order,final_order,dimension_subsystems):
    
    subsystems_number = len(initial_order)
    
    # check that order and dimension tuples have the same length
    if subsystems_number != len(final_order) or subsystems_number != len(dimension_subsystems):
        raise RuntimeError("The length of the tuples passed to the function needs to be the same") 
    
    # Create the list of basis for each subsystem
    initial_basis_list = list(map(lambda dim : basis(dim) , dimension_subsystems))

    # Create all possible indices for the global basis
    indices = indices_list(dimension_subsystems)

    # Create permutation matrix P
    total_dim = np.product(np.array(dimension_subsystems))            
    P_matrix = np.zeros((total_dim,total_dim))

    for index in indices:
        initial_vector_list = [initial_basis_list[n][i] for n,i in enumerate(index)]
        final_vector_list = [initial_vector_list[i] for i in final_order]
        initial_vector = tensor(initial_vector_list)
        final_vector = tensor(final_vector_list)

        P_matrix += np.outer(final_vector,initial_vector)

    return P_matrix

# This function maps a binary list into an integer
#
# NOTE: The function can be generalised to the case in which the list is non-binary.
#		We do so by using a different basis number.
#
#		Eg. Suppose we have a d-nary number (i_{n-1}, ... , i_0) where each i_* \in [0, ... , d-1]
#			Then the integer is given by
#
#			int_{i_{n-1}, ... , i_0} = sum_{m=0}^{n-1} i_m * d^m 
#
def binarytoint(binary_list,base=2):
    return np.sum([i*base**n for n,i in enumerate(reversed(binary_list))])

# This function maps a list (not necessary binary, each element can have different alphabet) into an integer
#
def seqtoint(sequence,dimension):
    for n,d in enumerate(dimension[1:]):
        sequence =[d*element for element in sequence[:n+1]] + list(sequence[n+1:])
    return int(sum(sequence))

# This function maps the cvx expression into a numpy array (0-1-2D)
#
# NOTE: This is going to be the part which slows down partial trace (due to double for loop)
#
# Function suggested by rileyjmurray on https://github.com/cvxgrp/cvxpy/issues/563
#
def expr_as_np_array(cvx_expr):
    if cvx_expr.is_scalar():
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1:
        return np.array([v for v in cvx_expr])
    else:
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i,j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr

#  This function maps a 2D numpy array into a cvx expression
#
# Function suggested by rileyjmurray on https://github.com/cvxgrp/cvxpy/issues/563
#
def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cp.bmat(aslist)
    return expr

# This decorator maps functions acting on numpy arrays to functions acting on cvxpy expression
#
# Decorating function suggested by rileyjmurray on https://github.com/cvxgrp/cvxpy/issues/563
#
def cvxify(f):

    @fc.wraps(f)
    def decorated(*args, **kwargs):

    	if not isinstance(args[0], Expression):
        	raise TypeError("The object passed is not a cvx Expression")
        
    	if len(args[0].shape) != 2:
        	raise ValueError("The object passed is not a 2D matrix")

    	rho = expr_as_np_array(args[0])
    	additional_args = args[1:]
    	f_rho = f(rho, *additional_args, **kwargs)
    	output_rho = np_array_as_expr(f_rho)

    	return output_rho

    return decorated

# This (decorated) function perform the partial trace over the subsystem defined by 'axis' of a 2D numpy matrix
#
# Inputs:
#		- rho: a (squared) matrix
#		- dims: a list containing the dimension of each subsystem
#		- axis: the index of the subsytem to be traced out
#
# Function suggested by dbunandar on https://github.com/cvxgrp/cvxpy/issues/563
#
def partial_trace_numpy(rho, dims, axis=0):
    
    dims_ = np.array(dims)
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)

    return traced_out_rho.reshape([rho_dim, rho_dim])

# This (decorated) function perform the partial trace over the subsystem defined by 'axis' of a 2D cvx matrix
#
partial_trace = cvxify(partial_trace_numpy)

# This (decorated) function implements partial transpose of a 2D cvx matrix with respect to a given 'mask'
#
# - rho: the 2D cvx variable
# - dims: tuple of dimensions of the subsystems
# - mask: tuple of 0/1, specifying which subsystem to transpose (1 = transpose, 0 = do not transpose)
#
# The implementation has been taken from QuTiP and readapted for numpy arrays.
#
@cvxify
def partial_transpose(rho, dims, mask):

    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, i] for n,i in enumerate(mask)],
                             [pt_dims[n, 1 - i] for n,i in enumerate(mask)]])
    pt_reshape = np.array([dims,dims]).flatten()
    pt_rho = rho.reshape(pt_reshape).transpose(pt_idx).reshape(rho.shape)

    return pt_rho

# This function implements the rule of the CHSH game
#
def CHSH_rule_function_A1Q1A2Q2(a1,q1,a2,q2):
    rule = (a1+a2)%2 == (q1 and q2)
    return rule

# This function implements the I3322 rule for the inequality
#
def general_I3322_rule_ineq(a,x,b,y,probQ1,probQ2):
    
    value = str(a)+str(b)+str(x)+str(y) # Note that the order here is different from input
    
    factors_abxy = {'0000':1.-2.*probQ1[x]-probQ2[y],
                    '0001':1.-probQ1[x]-probQ2[y],
                    '0002':1.-probQ2[y],
                    '0010':1.-2.*probQ1[x],
                    '0020':1.-2.*probQ1[x],
                    '0011':1.-probQ1[x],
                    '0012':-1.,
                    '0021':-1.-probQ1[x],
                    '0100':-probQ2[y],
                    '0101':-probQ2[y],
                    '0102':-probQ2[y],
                    '1000':-2.*probQ1[x],
                    '1001':-probQ1[x],
                    '1010':-2.*probQ1[x],
                    '1020':-2.*probQ1[x],
                    '1011':-probQ1[x],
                    '1021':-probQ1[x]
                   }
    
    if value in factors_abxy.keys():
        factor = factors_abxy[value]
        if np.isclose(factors_abxy[value], 0., atol=1e-10):
            return None
        else:
            return factor
    else:
        return None

# Produce random distribution for questions
#
def random_question_distribution(dim_question):
    unnorm_dist = np.random.rand(dim_question)
    norm = np.sum(unnorm_dist)
    return unnorm_dist/norm

# This function generates a random rule function and outputs it (together with the diagonal of the rule matrix)
#
def generate_random_rule_function_A1Q1A2Q2(subs_A1Q1A2Q2):
    
    dim_A1Q1A2Q2 = fc.reduce(mul, subs_A1Q1A2Q2, 1)
    V = np.random.randint(0, high=2, size=dim_A1Q1A2Q2)
    
    def random_rule_function_A1Q1A2Q2(a1,a2,q1,q2):
        rule = V[seqtoint((a1,a2,q1,q2),subs_A1Q1A2Q2)] == 1
        return rule
    
    return random_rule_function_A1Q1A2Q2, V

# Create rule function from list of values
#
def generate_rule_function_from_array(V,subs_A1Q1A2Q2):
    
    def rule_function_A1Q1A2Q2(a1,a2,q1,q2):
        rule = V[seqtoint((a1,a2,q1,q2),subs_A1Q1A2Q2)] == 1
        return rule
    
    return rule_function_A1Q1A2Q2

# This function implements the rule matrix from a rule function
#
# Inputs:
#		- dimensionAQ: tuple of dimensions of each answer and question space
#		- rule_function: function mapping the question/answer values to score (0,1)
#
def rule_matrix(dimensionAQ, rule_function):
    
    # Create the list of basis for each answer and question subspace
    basis_list = list(map(lambda dim : basis(dim) , dimensionAQ))
    
    # Create all possible indices for the global basis of answers and questions
    indices = indices_list(dimensionAQ)
    
    # Create the rule matrix
    total_dim = np.product(np.array(dimensionAQ))            
    V = np.zeros((total_dim,total_dim))
    
    for index in indices:
        if rule_function(*index) == True:
            vector_list = [basis_list[n][i] for n,i in enumerate(index)]
            basis_vector = tensor(vector_list)
            V += np.outer(basis_vector,basis_vector)
            
    return V

# This function reorders the index in the following way
#
# From (A1Q1)_1 ... (A1Q1)_n1 (A2Q2)_1 ... (A2Q2)_n2
# To   (A1_1 ... A1_n1)(Q1_1 ... Q1_n1)(A2_1 ... A2_n2)(Q2_1 ... Q2_n2)
#
#    INPUT:
#          - index: the numpy array with the index
#          - n1: extension for T
#          - n2: extension for \hat{T}
#
def reorder_index(index,n1,n2):
    indexA1 = index[0:2*n1-1:2]
    indexQ1 = index[1:2*n1:2]
    indexA2 = index[2*n1:2*(n1+n2)-1:2]
    indexQ2 = index[2*n1+1:2*(n1+n2):2]

    return np.concatenate((indexA1,indexQ1,indexA2,indexQ2))
    
# This function returns the n-th level of the classical hierarchy
#
# INPUT:
#        - constraints : list where we can append constraints
#        - subs_A1Q1 : tuple of dimension of A1 and Q1
#        - subs_A2Q2 : tuple of dimension of A2 and Q2
#        - probQ1 : tuple with the probability distribution of Q1
#        - probQ2 : tuple with the probability distribution of Q2
#        - level : the level of the hierarchy
#
# OUTPUT:
#        - classical_prob : variable representing the classical probability distribution
#        - BtI_ext : function for mapping from extended indices to integers
#
def classical_constraints(constraints,subs_A1Q1,subs_A2Q2,probQ1,probQ2,level=1):

    # Compute the dimension of the new variable
    dimA1, dimQ1 = subs_A1Q1
    dimA2, dimQ2 = subs_A2Q2
    dim_A1Q1 = fc.reduce(mul, subs_A1Q1, 1)
    dim_A2Q2 = fc.reduce(mul, subs_A2Q2, 1)
    dim_tot = dim_A1Q1*dim_A2Q2**level

    # Create the indices lists we need
    indices_A1Q1 = indices_list(subs_A1Q1)
    indices_A2Q2 = indices_list(subs_A2Q2)
    indices_A2Q2_extended = indices_list(subs_A2Q2*level)
    indices_A2Q2_ext_butone = indices_list(subs_A2Q2*(level-1))

    # Bit string to integer function for extended distribution
    BtI_ext = lambda seq : seqtoint(seq, subs_A1Q1+subs_A2Q2*level)

    # Create the classical variable (with non-negative entries)
    classical_prob = cp.Variable(dim_tot,nonneg=True)

    # Classical constraints for the given level of the hierarchy

    # i) The distribution is a proper probability distribution (elements adds up to 1)
    constraints.append( cp.sum(classical_prob) - 1 == 0 )

    # ii) Permutation invariance of A2Q2^level (the constraints we get here can be redundant)
    for index_A1Q1 in indices_A1Q1:
        for index_A2Q2_ext in indices_A2Q2_extended:
            index_A1Q1A2Q2_ext = np.append(index_A1Q1,index_A2Q2_ext)
            
            for i in range(level-1):
                index_A2Q2_ext_perm = np.copy(index_A2Q2_ext)
                
                # Re-arrange answers and questions
                index_A2Q2_ext_perm[2*i],index_A2Q2_ext_perm[2*(i+1)] = index_A2Q2_ext[2*(i+1)],index_A2Q2_ext[2*i]
                index_A2Q2_ext_perm[2*i+1],index_A2Q2_ext_perm[2*i+3] = index_A2Q2_ext[2*i+3],index_A2Q2_ext[2*i+1]
                
                index_A1Q1A2Q2_perm = np.append(index_A1Q1,index_A2Q2_ext_perm)

                rhs = classical_prob[BtI_ext(index_A1Q1A2Q2_ext)]
                lhs = classical_prob[BtI_ext(index_A1Q1A2Q2_perm)]
                constraints.append( rhs - lhs == 0 )

    # iii) Non-signalling A
    for q1 in range(dimQ1):
        for index_A2Q2_ext in indices_A2Q2_extended:
            indices_A1q1a2q2_ext = [np.append(np.array([a,q1]),index_A2Q2_ext) for a in range(dimA1)]
            indices_A1Q1a2q2_ext = [np.append(index_A1Q1,index_A2Q2_ext) for index_A1Q1 in indices_A1Q1]

            rhs = sum([classical_prob[BtI_ext(i)] for i in indices_A1q1a2q2_ext])
            lhs = probQ1[q1] * sum([classical_prob[BtI_ext(i)] for i in indices_A1Q1a2q2_ext])

            constraints.append( rhs - lhs == 0 )

    # iv) Non-signalling B
    indices_A1Q1A2Q2_ext_butone = fuse_arrays(indices_A1Q1,indices_A2Q2_ext_butone)

    for q2 in range(dimQ2):
        for index_A1Q1A2Q2_ext_butone in indices_A1Q1A2Q2_ext_butone:
            indices_a1q1A2q2_ext = [np.append(index_A1Q1A2Q2_ext_butone,np.array([a,q2])) for a in range(dimA2)]
            indices_a1q1A2Q2_ext = [np.append(index_A1Q1A2Q2_ext_butone,index_A2Q2) for index_A2Q2 in indices_A2Q2]

            rhs = sum([classical_prob[BtI_ext(i)] for i in indices_a1q1A2q2_ext])
            lhs = probQ2[q2] * sum([classical_prob[BtI_ext(i)] for i in indices_a1q1A2Q2_ext])

            constraints.append( rhs - lhs == 0 )
    
    return classical_prob, BtI_ext

# This function returns an operator encoding information on a given inequality
#
# See Appendix F in https://arxiv.org/abs/2005.08883 for more information
#
# INPUT:
#        - factors_ineq : function returning the factors multiplying p(a,b|x,y) in the inequality
#        - indices_A1Q1A2Q2 : list of all indices for a,b,x,y
#        - dimQ1 : dimension of questions for Alice
#        - dimQ2 : dimension of questions for Bob
#        - dimT : dimension of assisting quanutm system
#
# OUTPUT:
#        - I_op : a 2D matrix encoding the information on the inequality
#
def I_operator(factors_ineq,indices_A1Q1A2Q2,dimQ1,dimQ2,dimT):
    
    qubit_tot = 2+dimQ1+dimQ2
    subs_tot = tuple(np.full(qubit_tot, dimT))
    dim_tot = fc.reduce(mul, subs_tot, 1)

    # Initial order for permutations
    init_order = np.arange(qubit_tot)

    # Swap operator between two subsystems
    F_12 = permutation_matrix((0,1), (1,0), (dimT, dimT))

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

        M2 = tensor( [M1, Id] )
        N2 = tensor( [N1, Id] )

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
        P = permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
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
        P = permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
        N = P @ N2 @ P.T # Notice that N it is Identity over A A1 A2 A3 (also on all B* apart from By)

        # Use assumption on W state to reduce M and N operator
        known_W = tensor( [eigenproj_ZZ,Id] )

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
        P = permutation_matrix(tuple(init_order), tuple(final_order), subs_tot)
        Q = known_W @ P @ M @ N @ P.T

        # Reduce the dimension by tracing out the part of the operator we already know (last two subsystems)
        partial_dimension = [dimT**2,dimT**(qubit_tot-2)]
        I_list.append( factor * partial_trace_numpy(Q, partial_dimension) )
        
    I_op = np.sum(I_list,axis=0)
    
    return np.around(I_op)