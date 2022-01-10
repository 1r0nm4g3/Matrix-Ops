from matrix_ops.matrix_ops import Matrix
import pytest

@pytest.mark.parametrize("M1, M2, M3, result", [
    (Matrix([[1, 2, 3], [4, 5, 6]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[1, 1, 1], [1, 1, 1]], True),
    (Matrix([[0, 0, 0], [0, 0, 0]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[0, 0, 0], [0, 0, 0]], True),
    (Matrix([[7, 2, 4], [4, 8, 9]]), Matrix([[9, 3, 1], [4, 2, 9]]), [[0, 0, 0], [1, 0, 1]], True),
    (Matrix([[1, 2, 2], [2, 2, 3]]), 2, [[0, 1, 1], [1, 1, 0]], True),
    (Matrix([[1]]), 1, [[1]], True),
    (Matrix([[10, 22], [-3, 4]]), -3, [[0, 0], [1, 0]], True),
    (Matrix([[3, 2, 5], [9, 2, 4]]), Matrix([[8, 3, 5], [4, 2, 4]]), [[1, 1, 0], [1, 0, 1]], False)
])

def test_eq(M1, M2, M3, result):
    assert Matrix.true_equal((M1 == M2), Matrix(M3)) == result

@pytest.mark.parametrize("M1, M2, M3, result", [
    (Matrix([[1, 2, 3], [4, 5, 6]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[0, 0, 0], [0, 0, 0]], True),
    (Matrix([[0, 0, 0], [0, 0, 0]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[1, 1, 1], [1, 1, 1]], True),
    (Matrix([[7, 2, 4], [4, 8, 9]]), Matrix([[9, 3, 1], [4, 2, 9]]), [[1, 1, 0], [0, 0, 0]], True),
    (Matrix([[1, 2, 2], [2, 2, 3]]), 2, [[1, 0, 0], [0, 0, 0]], True),
    (Matrix([[1]]), 1, [[0]], True),
    (Matrix([[10, 22], [-3, 4]]), 4, [[0, 0], [1, 0]], True),
    (Matrix([[3, 2, 5], [9, 2, 4]]), Matrix([[8, 3, 5], [4, 2, 4]]), [[1, 1, 0], [1, 0, 1]], False)
])
def test_lt(M1, M2, M3, result):
    assert Matrix.true_equal((M1 < M2), Matrix(M3)) == result

@pytest.mark.parametrize("M1, M2, M3, result", [
    (Matrix([[1, 2, 3], [4, 5, 6]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[1, 1, 1], [1, 1, 1]], True),
    (Matrix([[0, 0, 0], [0, 0, 0]]), Matrix([[1, 2, 3], [4, 5, 6]]), [[1, 1, 1], [1, 1, 1]], True),
    (Matrix([[7, 2, 4], [4, 8, 9]]), Matrix([[9, 3, 1], [4, 2, 9]]), [[1, 1, 0], [1, 0, 1]], True),
    (Matrix([[1, 2, 2], [2, 2, 3]]), 2, [[1, 1, 1], [1, 1, 0]], True),
    (Matrix([[1]]), 1, [[1]], True),
    (Matrix([[10, 22], [-3, 4]]), 4, [[0, 0], [1, 1]], True),
    (Matrix([[3, 2, 5], [9, 2, 4]]), Matrix([[8, 3, 5], [4, 2, 4]]), [[1, 1, 0], [1, 0, 1]], False)
])
def test_le(M1, M2, M3, result):
    assert Matrix.true_equal((M1 <= M2), Matrix(M3)) == result

def test_zero_matrix():
    # Test that the output is the correct size.
    assert Matrix.zero_matrix(2,3) == [[0, 0, 0], [0, 0, 0]]

    # Deal with improper inputs. Must be non-negatve / zero ints/floats.
    with pytest.raises(Exception, match=r"Inputs must be ints or floats."):
        Matrix.zero_matrix(3, "h")
    with pytest.raises(Exception, match=r"Inputs must be greater than or equal to one."):
        Matrix.zero_matrix(0, 4)
    with pytest.raises(Exception, match=r"Inputs must be greater than or equal to one."):
        Matrix.zero_matrix(3, -2)

def test_verify_matrix():
    # Test cases of improper matrices
    with pytest.raises(Exception, match=r"All elements of a matrix must be either ints or floats"):
        Matrix.verify_matrix([[1, 3], [1, 3], ["r", 3]])
    with pytest.raises(Exception, match=r"A matrix must be a list, containing lists, which contain ints or floats."):
        Matrix.verify_matrix("pineapple")
    with pytest.raises(Exception, match=r"Each row of the matrix must contain the same number of elements."):
        Matrix.verify_matrix([[2, 3], [15, 20, 17], [0, 4], [17, 2]])
    with pytest.raises(Exception, match=r"A matrix must be a list, containing lists, which contain ints or floats."):
        Matrix.verify_matrix([[]])

@pytest.mark.parametrize("test_input, expected", [
    ([[1, 2, 3], [0, 4, 6]], (2, 3)),
    ([[2, 3], [1, 1], [0, 4]], (3, 2)),
    (Matrix([[2, 2], [3, 3]]), (2, 2)),
    (Matrix([[1]]), (1, 1))
])

def test_shape(test_input, expected):
    # Test that shape gives the correct output
    assert Matrix.shape(test_input) == expected

def test_dot_value():
    M1 = [[1, 2, 3], [0, 4, 6]]
    M2 = [[2, 3], [1, 1], [0, 4]]
    assert Matrix.dot(M1, M2) == [[4, 17], [4, 28]]

def test_dot_size():
    M1 = [[1, 2, 3], [0, 4, 6]]
    M3 = [[2, 3], [1, 1], [0, 4], [4, 7]]
    with pytest.raises(Exception, match=r"Shape Error:"):
        Matrix.dot(M1, M3)

def test_transpose():
    M = [[1, 2, 3], [0, 4, 6]]
    assert Matrix.transpose(M) == [[1, 0], [2, 4], [3, 6]]