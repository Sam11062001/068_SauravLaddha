#from google.colab import drive
#drive.mount("/content/drive")

# importing modules
import numpy as np
import pandas as pd

# creating and printing two matrics
arr1 = np.array([[5, 9, 3],[3, 7, 5]])
print(arr1)
arr2 = np.array([[3,-9],[5, 4],[7, -8]])
print(arr2)

# showing ho to randomly initialize an array
print(np.random.rand(3, 4))

# showing how to perform matrix multiplication
print(np.dot(arr1, arr2))

# showing how to perform element-wise matrix multiplication
res = [[0 for x in range(len(arr1))] for y in range(len(arr2[0]))]

for i in range(len(arr1)): 
  for j in range(len(arr2[0])): 
    for k in range(len(arr2)): 
      res[i][j] += arr1[i][k] * arr2[k][j]
 
print (res)


# showing how to find the mean of the first matrix
mean = np.mean(arr1)
print(mean)

main_data = pd.read_csv("mtcars.csv")
del main_data['model']

# shwing the meancenter
meancenter = main_data.apply(lambda e: e - e.mean())
print(meancenter.head())

