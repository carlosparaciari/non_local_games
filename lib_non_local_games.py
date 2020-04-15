import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
import functools as fc
from operator import mul
from itertools import permutations

# This function returns the computational basis of an Hilbert space with given dimension dim.
def basis(dim):
    basis = []
    for i in range(dim):
        vec = np.zeros(dim)
        vec[i] = 1
        basis.append(vec)
    return basis

# This function returns the tensor product of n states of different dimensions, passed as a list.
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
def fuse_arrays(v,w):
    
    if w.size != 0:
        fused_vw = np.empty((0,v.shape[1]+w.shape[1]))
        broadcast_shape = (w.shape[0],v.shape[1])

        for subv in v:
            broad_suv = np.broadcast_to(subv,broadcast_shape)
            fused_vw = np.vstack((fused_vw,np.hstack((broad_suv,w))))

        return fused_vw
    else:
        return v

# This function returns the unnormalized bipartite maximally entangled state of dimension dim.
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
def seqtoint(sequence,dimension):
    for n,d in enumerate(dimension[1:]):
        sequence =[d*element for element in sequence[:n+1]] + list(sequence[n+1:])
    return int(sum(sequence))

# This function maps the cvx expression into a numpy array (0-1-2D)
#
# NOTE: This is going to be the part which slows down partial trace (due to double for loop)
#
# Function suggested by rileyjmurray on https://github.com/cvxgrp/cvxpy/issues/563
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
def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cp.bmat(aslist)
    return expr

# This decorator maps functions acting on numpy arrays to functions acting on cvxpy expression
#
# Decorating function suggested by rileyjmurray on https://github.com/cvxgrp/cvxpy/issues/563
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

# This (decorated) function perform the partial trace over the subsystem defined by 'axis' of a 2D cvx matrix
#
# Inputs:
#		- rho: a (squared) matrix
#		- dims: a list containing the dimension of each subsystem
#		- axis: the index of the subsytem to be traced out
#
# Function suggested by dbunandar on https://github.com/cvxgrp/cvxpy/issues/563
@cvxify
def partial_trace(rho, dims, axis=0):
    
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

# This (decorated) function implements partial transpose of a 2D cvx matrix with respect to a given 'mask'
#
# - rho: the 2D cvx variable
# - dims: tuple of dimensions of the subsystems
# - mask: tuple of 0/1, specifying which subsystem to transpose (1 = transpose, 0 = do not transpose)
#
# The implementation has been taken from QuTiP and readapted for numpy arrays.
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
def CHSH_rule_function_A1Q1A2Q2(a1,q1,a2,q2):
    rule = (a1+a2)%2 == (q1 and q2)
    return rule

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

            # Create all possible permutations (of the blocks A2Q2)
            block_shape = (int(index_A2Q2_ext.size/2),2)
            index_A2Q2_block = np.reshape(index_A2Q2_ext,block_shape)
            indices_A2Q2_block = permutations(index_A2Q2_block)

            for index_A2Q2_block in indices_A2Q2_block:

                #Reshape back the A2Q2_ext from blocks to single array
                index_A2Q2_perm = np.reshape(index_A2Q2_block,index_A2Q2_ext.shape)

                if (index_A2Q2_perm == index_A2Q2_ext).all():
                    continue
                else:
                    index_A1Q1A2Q2_perm = np.append(index_A1Q1,index_A2Q2_perm)
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