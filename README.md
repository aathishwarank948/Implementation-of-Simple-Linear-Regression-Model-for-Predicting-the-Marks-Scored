# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Prepare the Data
Create/load the dataset (Hours_Studied as feature X, Marks_Scored as target y), then split it into training and testing sets (80% train, 20% test).

Step 2: Train the Model
Fit a Linear Regression model on the training data (X_train, y_train) to learn the best-fit line: y = b1·x + b0, computing the slope (b1) and intercept (b0).

Step 3: Predict and Evaluate
Use the trained model to predict marks on the test data (X_test), then evaluate performance using Mean Squared Error (MSE) and R² Score to check how well the line fits.

Step 4: Visualize and Apply
Plot the actual data points against the predicted regression line for a visual check, then use the trained model to predict marks for a new/unseen input (e.g., 7.5 hours studied).

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Aathishwaran K
RegisterNumber:  212225040006
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]]   
y = df["Marks_Scored"]      

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")

```


## Output:

<img width="652" height="352" alt="image" src="https://github.com/user-attachments/assets/356afcc7-4db4-4c20-997e-045eb66129a7" />

<img width="475" height="178" alt="image" src="https://github.com/user-attachments/assets/5d189b0e-2a3d-4848-8511-0c7ed8771746" />

<img width="1022" height="691" alt="image" src="https://github.com/user-attachments/assets/0a76f88c-b31d-454f-bef0-a9fe34575997" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.









