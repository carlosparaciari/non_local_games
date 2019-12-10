import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
import functools as fc

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

# This function returns the unnormalized bipartite maximally entangled state of dimension dim.
def bipartite_unnorm_max_entangled_state(dim):
	return sum([tensor([v,v]) for v in basis(dim)])

# This function returs the list of indices for a tuple of subsystem of given dimension
#
# NOTE: the output list is **not** ordered.
#
def indices_list(dimension_tuple):
    subindices = [range(dim) for dim in dimension_tuple]
    number_subsystems = len(dimension_tuple)
    return np.array(np.meshgrid(*subindices)).T.reshape(-1,number_subsystems)

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