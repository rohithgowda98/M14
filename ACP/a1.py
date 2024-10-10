import numpy as np

# Generate a 3x3 matrix 'matrix_a' with values from 0 to 8
matrix_a = np.arange(9, dtype=float).reshape(3, 3)
print("Matrix 'matrix_a':")
print(matrix_a)

# Create a 1D array 'array_b' with values [10, 10, 10]
array_b = np.array([10, 10, 10])
print("\nArray 'array_b':")
print(array_b)

# Element-wise addition of 'matrix_a' and 'array_b'
print("\nElement-wise addition of 'matrix_a' and 'array_b':")
print(np.add(matrix_a, array_b))

# Element-wise subtraction of 'array_b' from 'matrix_a'
print("\nElement-wise subtraction of 'array_b' from 'matrix_a':")
print(np.subtract(matrix_a, array_b))

# Element-wise multiplication of 'matrix_a' and 'array_b'
print("\nElement-wise multiplication of 'matrix_a' and 'array_b':")
print(np.multiply(matrix_a, array_b))

# Element-wise division of 'matrix_a' by 'array_b'
print("\nElement-wise division of 'matrix_a' by 'array_b':")
print(np.divide(matrix_a, array_b))
