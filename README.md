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

/*
Program to implement the linear regression using gradient descent.
Developed by: THEJASWINI D
RegisterNumber:  212223110059
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv("student_scores.csv")
print(dataset.head())
print(dataset.tail())
```
![Screenshot 2024-09-02 191843](https://github.com/user-attachments/assets/5c3fe383-b671-491a-a6cd-f1d28f8d82da)
```
dataset.info()
```
![Screenshot 2024-09-02 192011](https://github.com/user-attachments/assets/4d6c811e-737c-4e60-97c7-e1eb94ca2ad7)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![Screenshot 2024-09-02 192101](https://github.com/user-attachments/assets/46d6350f-aff1-4a04-9d0b-7de7af34ea8f)
```
X.shape
```
![Screenshot 2024-09-02 192137](https://github.com/user-attachments/assets/91ec5426-f639-4cfb-b74e-3e717dfc6216)
```
Y.shape
```
![Screenshot 2024-09-02 192213](https://github.com/user-attachments/assets/e6ab5ea7-6084-43fa-a468-ce478f8b514a)
```
m=0
c=0
L=0.001
epochs=5000
n=float(len(X))
error=[]
from os import XATTR_CREATE
for i in range(epochs):
  Y_pred=m*X+c
  D_m=(-2/n)*sum(X*(Y-Y_pred))
  D_c=(-2/n)*sum(Y-Y_pred)
  m=m-L*D_c
  c=c-L*D_c
  error.append(sum(Y-Y_pred)**2)
  print(m,c)
```
![Screenshot 2024-09-02 192402](https://github.com/user-attachments/assets/5f55d8b3-f5b2-4b1d-86f9-70d467c26b4b)
```
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
```
![Screenshot 2024-09-02 192453](https://github.com/user-attachments/assets/bb20beee-4273-46a8-bca1-1087212e7527)
![Screenshot 2024-09-02 192502](https://github.com/user-attachments/assets/fc0d6055-a096-4adc-96bf-b88d1e2f47ff)

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
