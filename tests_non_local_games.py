import lib_non_local_games as nlg
import numpy as np
import cvxpy as cp
from nose.tools import assert_raises, assert_equal

# ---------------- BASIS FUNCTION ----------------

def test_basis_zero_dimension():

	dim = 0

	expected_basis = []
	obtained_basis = nlg.basis(dim)

	assert_equal(obtained_basis, expected_basis)

def test_basis_non_zero_dimension():

	dim = 3

	expected_basis = [np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]
	obtained_basis = nlg.basis(dim)

	for exp_vec, obt_vector in zip(expected_basis,obtained_basis):
		np.testing.assert_allclose(exp_vec, obt_vector)

# ---------------- TENSOR FUNCTION ----------------

def test_tensor_one_vector():

	array_list = [np.array([1.,0.,0.])]
	
	expected_vector = np.array([1.,0.,0.])
	obtained_vector = nlg.tensor(array_list)

	np.testing.assert_allclose(expected_vector, obtained_vector)

def test_tensor_three_vector():

	array_list = [np.array([1.,0.]),np.array([0.,1.,0.]),np.array([0.,1.])]
	
	expected_vector = np.zeros(12)
	expected_vector[3] = 1.
	obtained_vector = nlg.tensor(array_list)

	np.testing.assert_allclose(expected_vector, obtained_vector)

def test_tensor_two_states():

	array_list = [np.array([[0.,1.],[0.5,0.]]),np.array([[1.,0.4],[2.,0.]])]
	
	expected_vector = np.array([[0.,0.,1.,0.4],[0.,0.,2.,0.],[0.5,0.2,0.,0.],[1.,0.,0.,0.]])
	obtained_vector = nlg.tensor(array_list)

	np.testing.assert_allclose(expected_vector, obtained_vector)

# ---------------- INDICES LIST FUNCTION ----------------

def test_indices_list_one_dim():

	dimension_tuple = (3,)

	expected_indices = np.array([[0],[1],[2]])
	obtained_indices = nlg.indices_list(dimension_tuple)

	np.testing.assert_allclose(expected_indices, obtained_indices)

def test_indices_list_multiple_dim():

	dimension_tuple = (2,3,2)

	# Notice that the list is not binary ordered 
	expected_indices = np.array([[0, 0, 0],
								 [0, 1, 0],
								 [0, 2, 0],
								 [1, 0, 0],
								 [1, 1, 0],
								 [1, 2, 0],
								 [0, 0, 1],
								 [0, 1, 1],
								 [0, 2, 1],
								 [1, 0, 1],
								 [1, 1, 1],
								 [1, 2, 1]
								]
							   )

	obtained_indices = nlg.indices_list(dimension_tuple)

	np.testing.assert_allclose(expected_indices, obtained_indices)

# ---------------- PERMUTATION MATRIX FUNCTION ----------------

def test_permutation_matrix_inconsistent_tuple_sizes():

	initial_order = (0,1,2)
	final_order = (0,2,1)
	dimension_subsystems = (2,2)

	with assert_raises(RuntimeError):
		nlg.permutation_matrix(initial_order,final_order,dimension_subsystems)

def test_permutation_matrix_three_subsystems():

	initial_order = (0,1,2)
	final_order = (0,2,1)
	dimension_subsystems = (2,2,3)

	expected_matrix = np.array([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
								[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
								[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
								[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
								[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
								[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
								[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
								[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
								[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
								[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]])

	obtained_matrix = nlg.permutation_matrix(initial_order,final_order,dimension_subsystems)

	np.testing.assert_allclose(expected_matrix, obtained_matrix)

# ---------------- BINARY TO INT FUNCTION ----------------

def test_binarytoint_base2():

	binary_list = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
	expected_list = [0,1,2,3,4,5,6,7]

	for bin_number, exp_number in zip(binary_list,expected_list):
		obtained_num = nlg.binarytoint(bin_number)
		assert_equal(exp_number,obtained_num)

def test_binarytoint_base3():

	binary_list = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
	expected_list = [0,1,2,3,4,5,6,7,8]

	for bin_number, exp_number in zip(binary_list,expected_list):
		obtained_num = nlg.binarytoint(bin_number,base=3)
		assert_equal(exp_number,obtained_num)

# ---------------- CVXPY TO NUMPY FUNCTION ----------------

def test_expr_as_np_array_0D():

	input_0D = cp.Constant(3.)
	output_0D = nlg.expr_as_np_array(input_0D)

	expected_value = 3.
	obtained_value = output_0D.tolist().value

	assert_equal(expected_value, obtained_value)

def test_expr_as_np_array_1D():

	input_1D = cp.Constant([3.,1.,0.2])
	output_1D = nlg.expr_as_np_array(input_1D)

	expected_values = [3.,1.,0.2]
	obtained_values = output_1D.tolist()

	for exp_val, obt_val in zip(expected_values,obtained_values):
		assert_equal(exp_val, obt_val.value) 

def test_expr_as_np_array_2D():

	# CVXPY is rotating the row and column when loading inside a constant.
	# That's why we use cvxpy.bmat() in this test.
	input_2D = cp.bmat([[3.,1.,0.2],[0.1,5.,2.]])
	output_2D = nlg.expr_as_np_array(input_2D)

	expected_value = [[3.,1.,0.2],[0.1,5.,2.]]
	obtained_value = output_2D.tolist()

	for i in range(2):
		for j in range(3):
			assert_equal(expected_value[i][j], obtained_value[i][j].value) 
	
# ---------------- NUMPY TO CVXPY FUNCTION ----------------

def test_np_array_as_expr_2D():

	input_2D = np.array([[1.,3.,4.],[2.,6.,7.]])
	output_2D = nlg.np_array_as_expr(input_2D)

	np.testing.assert_allclose(input_2D, output_2D.value)

# ---------------- PARTIAL TRACE FUNCTION ----------------

rho_A = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
rho_B = np.random.rand(3, 3) + 1j*np.random.rand(3, 3)
rho_C = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)

rho_A /= np.trace(rho_A)
rho_B /= np.trace(rho_B)
rho_C /= np.trace(rho_C)

rho_AB = np.kron(rho_A, rho_B)
rho_BC = np.kron(rho_B, rho_C)
rho_AC = np.kron(rho_A, rho_C)
rho_ABC = np.kron(rho_AB, rho_C)

def test_partial_trace_no_cvx_expression():

	wrong_input = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])

	with assert_raises(TypeError):
		nlg.partial_trace(wrong_input, [3])

def test_partial_trace_no_squared_matrix():

	wrong_input = cp.bmat([[1.,2.,3.],[4.,5.,6.]])

	with assert_raises(ValueError):
		nlg.partial_trace(wrong_input, [3])

def test_partial_trace():

	cvx_rho_ABC = cp.Variable(shape=rho_ABC.shape, complex=True)
	cvx_rho_AB = cp.Variable(shape=rho_AB.shape, complex=True)
	cvx_rho_AC = cp.Variable(shape=rho_AC.shape, complex=True)

	cvx_rho_ABC.value = rho_ABC
	cvx_rho_AB.value = rho_AB
	cvx_rho_AC.value = rho_AC

	obtained_rho_AB = nlg.partial_trace(cvx_rho_ABC, [4, 3, 2], axis=2)
	obtained_rho_AC = nlg.partial_trace(cvx_rho_ABC, [4, 3, 2], axis=1)

	obtained_rho_A = nlg.partial_trace(cvx_rho_AB, [4, 3], axis=1)
	obtained_rho_B = nlg.partial_trace(cvx_rho_AB, [4, 3])
	obtained_rho_C = nlg.partial_trace(cvx_rho_AC, [4, 2])

	np.testing.assert_allclose(obtained_rho_AB.value, rho_AB)
	np.testing.assert_allclose(obtained_rho_AC.value, rho_AC)

	np.testing.assert_allclose(obtained_rho_A.value, rho_A)
	np.testing.assert_allclose(obtained_rho_B.value, rho_B)
	np.testing.assert_allclose(obtained_rho_C.value, rho_C)

# ---------------- PARTIAL TRANSPOSE FUNCTION ----------------

sigma_A = np.array([[1,0],[0,0]])
sigma_BC = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])
sigma_BC_TB = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
sigma_ABC = np.kron(sigma_A,sigma_BC)
sigma_ABC_TB = np.kron(sigma_A,sigma_BC_TB)

def test_partial_transpose_BC():

	cvx_sigma_BC = cp.Variable(shape=sigma_BC.shape)
	cvx_sigma_BC.value = sigma_BC

	obtained_sigma_BC_TB = nlg.partial_transpose(cvx_sigma_BC, [2,2], (1,0))
	obtained_sigma_BC_TC = nlg.partial_transpose(cvx_sigma_BC, [2,2], (0,1))

	np.testing.assert_allclose(obtained_sigma_BC_TB.value, sigma_BC_TB)
	np.testing.assert_allclose(obtained_sigma_BC_TC.value, sigma_BC_TB)

def test_partial_transpose_ABC_as_3_subsystems():

	cvx_sigma_ABC = cp.Variable(shape=sigma_ABC.shape)
	cvx_sigma_ABC.value = sigma_ABC

	obtained_sigma_ABC_TA = nlg.partial_transpose(cvx_sigma_ABC, [2,2,2], (1,0,0))
	obtained_sigma_ABC_TB = nlg.partial_transpose(cvx_sigma_ABC, [2,2,2], (0,1,0))
	obtained_sigma_ABC_TC = nlg.partial_transpose(cvx_sigma_ABC, [2,2,2], (0,0,1))

	np.testing.assert_allclose(obtained_sigma_ABC_TA.value, sigma_ABC)
	np.testing.assert_allclose(obtained_sigma_ABC_TB.value, sigma_ABC_TB)
	np.testing.assert_allclose(obtained_sigma_ABC_TC.value, sigma_ABC_TB)

def test_partial_transpose_ABC_as_2_subsystems_A_BC():

	cvx_sigma_ABC = cp.Variable(shape=sigma_ABC.shape)
	cvx_sigma_ABC.value = sigma_ABC

	obtained_sigma_ABC_TA = nlg.partial_transpose(cvx_sigma_ABC, [2,4], (1,0))
	obtained_sigma_ABC_TBC = nlg.partial_transpose(cvx_sigma_ABC, [2,4], (0,1))

	np.testing.assert_allclose(obtained_sigma_ABC_TA.value, sigma_ABC)
	np.testing.assert_allclose(obtained_sigma_ABC_TBC.value, sigma_ABC)

def test_partial_transpose_ABC_as_2_subsystems_AB_C():

	cvx_sigma_ABC = cp.Variable(shape=sigma_ABC.shape)
	cvx_sigma_ABC.value = sigma_ABC

	obtained_sigma_ABC_TAB = nlg.partial_transpose(cvx_sigma_ABC, [4,2], (1,0))
	obtained_sigma_ABC_TC = nlg.partial_transpose(cvx_sigma_ABC, [4,2], (0,1))

	np.testing.assert_allclose(obtained_sigma_ABC_TAB.value, sigma_ABC_TB)
	np.testing.assert_allclose(obtained_sigma_ABC_TC.value, sigma_ABC_TB)

# ---------------- CHSH RULE FUNCTION ----------------

def test_CHSH_rule_function():

	input_AQ = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
						 [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
						 [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
						 [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
					   )

	expected_output = np.array([True,True,False,False,True,False,False,True,
								False,False,True,True,False,True,True,False]
							  )

	for inp, exp_out in zip(input_AQ,expected_output):
		obt_out = nlg.CHSH_rule_function_A1Q1A2Q2(*inp)
		assert_equal(exp_out,obt_out)

# ---------------- RULE MATRIX FUNCTION ----------------

def test_rule_matrix_1A_1Q():

	dimensionAQ = (3,3)
	rule = lambda a,q : a == (q+1)%3

	expected_matrix_AQ = np.zeros((9,9))
	expected_matrix_AQ[2,2] = 1
	expected_matrix_AQ[3,3] = 1
	expected_matrix_AQ[7,7] = 1

	obtained_matrix_AQ = nlg.rule_matrix(dimensionAQ, rule)
	np.testing.assert_allclose(obtained_matrix_AQ, expected_matrix_AQ)

def test_rule_matrix_2A_2Q():

	dimensionAQ = (2,2,2,2)
	rule = lambda a1,a2,q1,q2 : (a1+a2)%2 == (q1 or q2)

	expected_matrix_AQ = np.zeros((16,16))
	expected_matrix_AQ[0,0] = 1
	expected_matrix_AQ[5,5] = 1
	expected_matrix_AQ[6,6] = 1
	expected_matrix_AQ[7,7] = 1
	expected_matrix_AQ[9,9] = 1
	expected_matrix_AQ[10,10] = 1
	expected_matrix_AQ[11,11] = 1
	expected_matrix_AQ[12,12] = 1

	obtained_matrix_AQ = nlg.rule_matrix(dimensionAQ, rule)
	np.testing.assert_allclose(obtained_matrix_AQ, expected_matrix_AQ)