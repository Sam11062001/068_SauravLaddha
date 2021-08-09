#from google.colab import drive
#drive.mount("/content/drive")
# Step 1: Import Libraries

import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Step 2: Load Data
        
datasets = pd.read_csv('Data_for_Transformation.csv') 
print("\nData :\n",datasets)
#print("\nData statistics\n",datasets.describe())

# Step 3: Seprate Input and Output attributes

# All rows, all columns except last 
X = datasets.iloc[:, :-1].values 
  
# Only last column  
Y = datasets.iloc[:, -1].values 

#print("\n\nInput : \n", X) 
#print("\n\nOutput: \n", Y) 

X_new = datasets.iloc[:,1:3].values
print("\n\nX for transformation : \n", X_new)

# Step 4 : Perform scaling on age and salary

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_new)
print("\n\nScaled X : \n", X_scaled)

# Step 5 : Perform standardization on age and salary

std = StandardScaler()
X_std = std.fit_transform(X_new)
print("\n\nStandardized X : \n", X_std)

