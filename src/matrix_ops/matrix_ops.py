from __future__ import annotations
from typing import List, Tuple, Union
import operator

# Matrix = List[List[float]]


class Matrix:
    def __init__(self, matrix):
        Matrix.verify_matrix(matrix)
        self.shape = Matrix.shape(matrix)
        self.value = matrix

    def __getitem__(self, x):
        return self.value[x]

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def logical_eval(M1: Matrix, comparison, M2: Union[Matrix, List, int, float]):
        if isinstance(M2, List):
            M2 = Matrix(M2)

        if isinstance(M2, Matrix):
            Matrix.compare_matrix_size(M1, M2)

            output = Matrix.zero_matrix(M1.shape[0], M1.shape[1])

            for row in range(M1.shape[0]):
                for element in range(M1.shape[1]):
                    if comparison(M1[row][element], M2[row][element]):
                        output[row][element] = 1

            return output

        if isinstance(M2, (int, float)):
            output = Matrix.zero_matrix(M1.shape[0], M1.shape[1])

            for row in range(M1.shape[0]):
                for element in range(M1.shape[1]):
                    if comparison(M1[row][element], M2):
                        output[row][element] = 1

            return output

    @staticmethod
    def math_eval(M1: Matrix, operation, M2: Matrix):
        if isinstance(M2, Matrix):
            Matrix.compare_matrix_size(M1, M2)

            output = Matrix.zero_matrix(M1.shape[0], M1.shape[1])

            for row in range(M1.shape[0]):
                for element in range(M1.shape[1]):
                    output[row][element] = operation(M1[row][element], M2[row][element])

            return output

        if isinstance(M2, (int, float)):
            output = Matrix.zero_matrix(M1.shape[0], M1.shape[1])

            for row in range(M1.shape[0]):
                for element in range(M1.shape[1]):
                    output[row][element] = operation(M1[row][element], M2)

            return output

    def __eq__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.eq, M2)

    def __lt__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.lt, M2)

    def __le__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.le, M2)

    def __ge__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.ge, M2)

    def __gt__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.gt, M2)

    def __ne__(self, M2: Matrix) -> Matrix:
        return self.logical_eval(self, operator.ne, M2)

    def __add__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.add, M2)

    def __sub__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.sub, M2)

    def __mul__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.mul, M2)

    def __div__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.div, M2)

    def __floordiv__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.floordiv, M2)

    def __neg__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.neg, M2)

    def __or__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.or_, M2)

    def __and__(self, M2: Matrix) -> Matrix:
        return self.math_eval(self, operator.and_, M2)

    def __abs__(self):
        output = Matrix.zero_matrix(self.shape[0], self.shape[1])
        for row in range(self.shape[0]):
            for element in range(self.shape[1]):
                output[row][element] = abs(self[row][element])
        return output

    @staticmethod
    def dot(M1: Matrix, M2: Matrix) -> Matrix:
        """
        Returns the dot product of two matrices.

        M1, M2: Matrices to be multiplied using the dot product

        output: Matrix, the dot product of the two matrices
        """
        Matrix.verify_matrix(M1)
        Matrix.verify_matrix(M2)

        M1_shape = Matrix.shape(M1)
        M2_shape = Matrix.shape(M2)

        if (M1_shape[1] != M2_shape[0]):
            raise Exception(f"Shape Error: dim M1 columns ({M1_shape[1]}) not equal to M2 rows ({M2_shape[0]})")

        output = Matrix.zero_matrix(M1_shape[0], M2_shape[1])
        try:
            for i in range(M1_shape[0]):
                for j in range(M2_shape[1]):
                    output[i][j] = sum([M1[i][k]*M2[k][j] for k in range(M1_shape[1])])
        except TypeError:
            print("fff")

        return output

    @staticmethod
    def zero_matrix(h: int, w: int) -> Matrix:
        """
        Creates a matrix of zeroes.

        h: int - height, # of rows in matrix
        w: int - width,  # of columns in matrix

        output: Matrix - h lists each containing a list of w zeores.
        """
        if not(isinstance(h, (int, float)) and isinstance(w, (int, float))):
            raise Exception("Inputs must be ints or floats.")

        if h < 1 or w < 1:
            raise Exception("Inputs must be greater than or equal to one.")

        output = []
        for i in range(int(h)):
            temp = []
            for j in range(int(w)):
                temp.append(0)
            output.append(temp)
        return Matrix(output)

    @staticmethod
    def compare_matrix_size(M1: Matrix, M2: Matrix) -> bool:
        if Matrix.shape(M1) != Matrix.shape(M2):
            raise Exception(f"Matrices must match size. M1 is {Matrix.shape(M1)} while M2 is {Matrix.shape(M2)}")

        pass

    @staticmethod
    def verify_matrix(M: Union[Matrix, List[List[Union[int, float]]]]) -> None:
        """
        Verifies that a given input follows the rules for a matrix by checking:

            1. The matrix is a list and contains lists.
            2. Each row is the same length
            3. All elements are either ints or floats

        Does not return anything.

        M: Matrix
        """
        if isinstance(M, list):
            if len(M) > 0:
                if isinstance(M[0], list):
                    if len(M[0]) < 1:
                        raise Exception("A matrix must be a list, containing lists, which contain ints or floats.")
                else:
                    raise Exception("A matrix must be a list, containing lists, which contain ints or floats.")
            else:
                raise Exception("A matrix must be a list, containing lists, which contain ints or floats.")
        else:
            if not(isinstance(M, Matrix)):
                raise Exception("A matrix must be a list, containing lists, which contain ints or floats.")
     
        for row in M:
            if len(row) != len(M[0]):
                raise Exception("Each row of the matrix must contain the same number of elements.")
            for element in row:
                if not(isinstance(element, int) or isinstance(element, float)):
                    raise Exception("All elements of a matrix must be either ints or floats.")
        pass

    @staticmethod
    def shape(M: Union[Matrix, list[list[Union[int, float]]]]) -> Tuple[int, int]:
        """
        Returns the number of rows and columns in a given matrix, M. Assumes the matrix given is a proper matrix.

        M: Matrix

        output: (rows or height, columns or width)
        """
        if isinstance(M, Matrix):
            return M.shape
        return (len(M), len(M[0]))

    @staticmethod
    def transpose(M: Matrix) -> Matrix:
        """
        Returns a transposed matrix.

        M: Matrix

        new: New matrix with transposed values
        """
        M_shape = Matrix.shape(M)
        new = Matrix.zero_matrix(M_shape[1], M_shape[0])
        for i in range(M_shape[0]):
            for j in range(M_shape[1]):
                new[j][i] = M[i][j]
        return new


M1 = Matrix([[1, 2, 3], [0, 4, 6]])
M2 = Matrix([[1, 2, 4], [4, 5, 6]])
M3 = Matrix([[-2, -3, -4], [-5, 12, 14]])
M4 = Matrix([[2, 3], [1, 1], [0, 4]])
