# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Get the independent variable X and dependent variable Y.
2.  Calculate the mean of the X -values and the mean of the Y -values.
3.  Find the slope m of the line of best fit using the formula. 
4.  Obtain the straight line equation Y=mX+b and plot the scatterplot.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Aathishwaran K
RegisterNumber:  212225040006
*/
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Create Dataset (Hours studied vs Marks scored)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored": [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df


# Step 3: Split into Features and Target
X = df[["Hours_Studied"]]
y = df["Marks_Scored"]
