# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: THEJASWINI D
RegisterNumber:  212223110059
*/
/*
Program to implement the linear regression using gradient descent.
Developed by: MERCY A
RegisterNumber: 212223110027 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)
```
![exp 3 -1](https://github.com/user-attachments/assets/e741039b-50cd-48f3-af42-ef41c10802a3)
```
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
![exp3-2](https://github.com/user-attachments/assets/9366b04c-5507-4edb-ac74-84d7a1476fee)
```
X1_Scaled=scaler.fit_transform(X1)
print(X1_Scaled)
```
![exp3-3](https://github.com/user-attachments/assets/36250bb4-487a-4271-8158-244502891e60)
```
Y1_Scaled=scaler.fit_transform(y)
print(Y1_Scaled)
```
![exp3-4](https://github.com/user-attachments/assets/50cf7306-8853-4e69-a0e7-60f2db2f8928)

## Output:
![exp3-5](https://github.com/user-attachments/assets/a9df90a8-539b-495e-b5ac-e117b8778c5d)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
