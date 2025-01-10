import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("adultData.csv")
print(data)
data['Sector'].replace(['Never-worked','Without-pay','Self-emp-not-inc','Self-emp-inc','Local-gov','State-gov','Federal-gov','Private'],[0,1,2,3,4,5,6,7],inplace=True)
data['Marital Status'].replace(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],[0,1,2,3,4,5,6],inplace=True)
data['Race'].replace(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],[0,1,2,3,4],inplace=True)
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)
data['Momnee?'].replace(['>50K', '<=50K'],[0,1],inplace=True)
x=data[['Age','Sector','EdNum','Marital Status','Race','Gender','Assists']].values
y=data["Momnee?"].values
print(x)
print(y)