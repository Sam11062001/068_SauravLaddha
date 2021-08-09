import numpy as np

alist = [1, 2, 3, 4, 5]                 # Define a python list. It looks like an np array
narray = np.array([1, 2, 3, 4])         # Define a numpy array

print(alist)
print(narray)
print(type(alist))
print(type(narray))

print(narray + narray)
print(alist + alist)

print(narray * 3)
print(alist * 3)

npmatrix1 = np.array([narray, narray, narray])          # Matrix initialized with NumPy arrays
npmatrix2 = np.array([alist, alist, alist])             # Matrix initialized with lists
npmatrix3 = np.array([narray, [1, 1, 1, 1], narray])    # Matrix initialized with both types
print(npmatrix1)
print(npmatrix2)
print(npmatrix3)

# Example 1:
okmatrix = np.array([[1, 2], [3, 4]])               # Define a 2x2 matrix
print(okmatrix)                                     # Print okmatrix
print(okmatrix * 2)                                 # Print a scaled version of okmatrix

# Example 2:
badmatrix = np.array([[1, 2], [3, 4], [5, 6, 7]])   # Define a matrix. Note the third row contains 3 elements
print(badmatrix)                                    # Print the malformed matrix
print(badmatrix * 2)                                # It is supposed to scale the whole matrix

# Scale by 2 and translate 1 unit the matrix
result = okmatrix * 2 + 1                           # For each element in the matrix, multiply by 2 and add 1
print(result)

# Add two sum compatible matrices
result1 = okmatrix + okmatrix
print(result1)
# Subtract two sum compatible matrices. This is called the difference vector
result2 = okmatrix - okmatrix
print(result2)

result = okmatrix * okmatrix                    # Multiply each element by itself
print(result)

matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])  # Define a 3x2 matrix
print('Original matrix 3 x 2')
print(matrix3x2)
print('Transposed matrix 2 x 3')
print(matrix3x2.T)

nparray = np.array([1, 2, 3, 4])                # Define an array
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

nparray = np.array([[1, 2, 3, 4]])              # Define a 1 x 4 matrix. Note the 2 level of square brackets
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

nparray1 = np.array([1, 2, 3, 4])               # Define an array
norm1 = np.linalg.norm(nparray1)
nparray2 = np.array([[1, 2], [3, 4]])           # Define a 2 x 2 matrix. Note the 2 level of square brackets
norm2 = np.linalg.norm(nparray2)
print(norm1)
print(norm2)

nparray2 = np.array([[1, 1], [2, 2], [3, 3]])   # Define a 3 x 2 matrix.
normByCols = np.linalg.norm(nparray2, axis=0)   # Get the norm for each column. Returns 2 elements
normByRows = np.linalg.norm(nparray2, axis=1)   # get the norm for each row. Returns 3 elements
print(normByCols)
print(normByRows)

nparray1 = np.array([0, 1, 2, 3])       # Define an array
nparray2 = np.array([4, 5, 6, 7])       # Define an array
flavor1 = np.dot(nparray1, nparray2)    # Way-1
print(flavor1)
flavor2 = np.sum(nparray1 * nparray2)   # Way-2
print(flavor2)
flavor3 = nparray1 @ nparray2           # Way-3
print(flavor3)
# As you never should do: #Way-4
flavor4 = 0
for a, b in zip(nparray1, nparray2):
    flavor4 += a * b
print(flavor4)

norm1 = np.dot(np.array([1, 2]), np.array([3, 4]))              # Dot product on nparrays
norm2 = np.dot([1, 2], [3, 4])                                  # Dot product on python lists
print(norm1, '=', norm2 )

nparray2 = np.array([[1, -1], [2, -2], [3, -3]])                # Define a 3 x 2 matrix.
sumByCols = np.sum(nparray2, axis=0)                            # Get the sum for each column. Returns 2 elements
sumByRows = np.sum(nparray2, axis=1)                            # get the sum for each row. Returns 3 elements
print('Sum by columns: ')
print(sumByCols)
print('Sum by rows:')
print(sumByRows)

nparray2 = np.array([[1, -1], [2, -2], [3, -3]])                # Define a 3 x 2 matrix. Chosen to be a matrix with 0 mean
mean = np.mean(nparray2)                                        # Get the mean for the whole matrix
meanByCols = np.mean(nparray2, axis=0)                          # Get the mean for each column. Returns 2 elements
meanByRows = np.mean(nparray2, axis=1)                          # get the mean for each row. Returns 3 elements
print('Matrix mean: ')
print(mean)
print('Mean by columns: ')
print(meanByCols)
print('Mean by rows:')
print(meanByRows)

nparray2 = np.array([[1, 1], [2, 2], [3, 3]])                   # Define a 3 x 2 matrix.
nparrayCentered = nparray2 - np.mean(nparray2, axis=0)          # Remove the mean for each column
print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)
print('New mean by column')
print(nparrayCentered.mean(axis=0))

nparray2 = np.array([[1, 3], [2, 4], [3, 5]])                   # Define a 3 x 2 matrix.
nparrayCentered = nparray2.T - np.mean(nparray2, axis=1)        # Remove the mean for each row
nparrayCentered = nparrayCentered.T                             # Transpose back the result
print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)
print('New mean by rows')
print(nparrayCentered.mean(axis=1))

