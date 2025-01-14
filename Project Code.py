import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("adultData.csv")
print(data)
data['Sector'].replace(['Never-worked','Without-pay','Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Private'],[0,1,2,3,4,5,6,7],inplace=True)
data['MaritalStatus'].replace(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],[0,1,2,3,4,5,6],inplace=True)
data['Race'].replace(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],[0,1,2,3,4],inplace=True)
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)
data['Money'].replace(['>50K', '<=50K'],[0,1],inplace=True)
x=data[['Age','Sector','EdNum','MaritalStatus','Race','Gender','Assists']].values
y=data["Money"].values
print(x)
print(y)
test = pd.read_csv("adultTest.csv")
print(test)
test['Sector'].replace(['Never-worked','Without-pay','Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Private'],[0,1,2,3,4,5,6,7],inplace=True)
test['MaritalStatus'].replace(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],[0,1,2,3,4,5,6],inplace=True)
test['Race'].replace(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],[0,1,2,3,4],inplace=True)
test['Gender'].replace(['Male','Female'],[0,1],inplace=True)
test['Money'].replace(['>50K', '<=50K'],[0,1],inplace=True)
x_test = test[['Age','Sector','EdNum','MaritalStatus','Race','Gender','Assists']].values
y_test = test["Money"].values
scaler = StandardScaler().fit(x)
# x = scaler.transform(x)
# model = linear_model.LogisticRegression().fit(x, y)
# print(f"Model Accuracy: {model.score(x, y)}")
# print("Testing Data:")
# for index in range(len(x_test)):
#     x = x_test[index]
#     x = x.reshape(-1, 3)
#     print(x)
#     y_pred = int(model.predict(x))

#     if y_pred == 0:
#         y_pred = "Male"
#     elif y_pred == 1:
#         y_pred = "Female"
    
#     actual = y_test[index]
#     if actual == 0:
#         actual = "Male"
#     elif actual == 1:
#         actual = "Female"
#     print(f"Predicted Gender: {y_pred} Actual Gender: {actual}")
#     print("")