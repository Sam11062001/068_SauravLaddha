#from google.colab import drive
#drive.mount("/content/drive")

# importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# Reading Data from csv file
main_data = pd.read_csv("Exercise-CarData.csv")
print("\nMain Data :-\n", main_data)

print("\nMain Data statistics:-\n", main_data.describe())

# Seprating Input and Output attributes from main data
X = main_data.iloc[:, :-1].values 
Y = main_data.iloc[:, -1].values 

print("\n\nInput Data:- \n", X) 
print("\n\nOutput Data:- \n", Y) 

X_new = main_data.iloc[:, 2:6].values
print("\n\nX for transformation :- \n", X_new)


# processing columns which are unused.
from sklearn.impute import SimpleImputer

# using the sklearn.impute => SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, [2, 6]])
print(f"\nImputer :- {imputer}")

# Transforming data for columns 2 to 6
X[:, [2, 6]] = imputer.transform(X[:, [2, 6]])
print(f"\n\nX is :- {X}")

imputer = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

imputer = imputer.fit(X[:, [4, 9]])
print(f"\n\nImputer :- {imputer}")

# Transforming data for columns 4 to 9
X[:, [4, 9]] = imputer.transform(X[:, [4, 9]])
print(f"\n\nX is :- {X}") 


# MinMaxScaler and StandardScaler Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X_temp = X[:, [1, 2, 6, 7, 8]]

# building scaler object
scaler = MinMaxScaler()

# scaling x_temp
scaled_x = scaler.fit_transform(X_temp)
print(f"\n\nScaled _x :- {scaled_x}")

# doing standard scaler
stad = StandardScaler()
std_x = stad.fit_transform(scaled_x)
print(f"\n\nstandard x :- {std_x}")


# Labeling
from sklearn import preprocessing

label_encode = preprocessing.LabelEncoder()
X[ : , 4] = label_encode.fit_transform(X[ : ,4])
X[ : , 9] = label_encode.fit_transform(X[ : ,9])

# dummy columns for Doors
duplicate_data = pd.get_dummies(main_data['Doors'])
print(f"\n\nDuplicate Data :- {duplicate_data}")
main_data = main_data.drop(['Doors', ], axis = 1)
main_data = pd.concat([duplicate_data, main_data], axis = 1)

# dummy columns for fuel type
diplicate_data = pd.get_dummies(main_data['FuelType'])
print(f"\n\nDuplicate Data :- {duplicate_data}")
main_data = main_data.drop(['FuelType',], axis = 1)
main_data = pd.concat([duplicate_data, main_data], axis = 1)

print("\n\nLast Data Heads \n",main_data.head(20))
print("\n\nLast Data Tails \n",main_data.tail(20))




import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt

data_corr = pd.read_csv('Exercise-CarData.csv')

print(f"\n\nData :- {data_corr}\n\n")

corr = data_corr.corr()
corr.head()

linewidths = 2
linecolor = "yellow"

sns.heatmap(corr)
plt.show()
sns.heatmap(corr, linewidths = linewidths, linecolor = linecolor)
plt.show()
sns.heatmap(corr, annot = True)
plt.show()

columns = np.full((corr.shape[0], ), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if (corr.iloc[i, j] >= 0.9):
            if (columns[j]):
                columns[j] = False

print(f"\n\nColumns are {columns}\n\n")

