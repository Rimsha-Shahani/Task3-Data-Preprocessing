# Task3-Data-Preprocessing
Data Preprocessing and Feature Engineering
# Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Create Dataset
data = {
    'Client_ID':[1,2,3,4,5,5],
    'Age':[25,30,np.nan,40,35,35],
    'Gender':['Male','Female','Female','Male',np.nan,np.nan],
    'City':['Karachi','Lahore','Islamabad','Karachi','Lahore','Lahore'],
    'Income':[50000,60000,55000,np.nan,70000,70000],
    'Purchased':['Yes','No','Yes','No','Yes','Yes']
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATASET =====")
print(df)

# Check Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Income'] = df['Income'].fillna(df['Income'].mean())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

print("\n===== AFTER HANDLING MISSING VALUES =====")
print(df)

# Remove Duplicates
df = df.drop_duplicates()

print("\n===== AFTER REMOVING DUPLICATES =====")
print(df)

# Label Encoding
le = LabelEncoder()
df['Purchased'] = le.fit_transform(df['Purchased'])

print("\n===== AFTER LABEL ENCODING =====")
print(df)

# One Hot Encoding
df = pd.get_dummies(df, columns=['City'])

print("\n===== AFTER ONE HOT ENCODING =====")
print(df)

# Min-Max Scaling
scaler = MinMaxScaler()
df[['Age','Income']] = scaler.fit_transform(df[['Age','Income']])

print("\n===== AFTER MINMAX SCALING =====")
print(df)

# Feature Selection
X = df.drop('Purchased', axis=1)
y = df['Purchased']

selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)

print("\n===== SELECTED FEATURES =====")
print(X.columns[selector.get_support()])

# Save Cleaned Dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\n===== DATASET SAVED SUCCESSFULLY =====")
print("File Name: cleaned_dataset.csv")
