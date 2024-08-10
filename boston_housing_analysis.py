import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#load the data set
file_path = "BostonHousingModel/Boston Housing Data.csv"
df = pd.read_csv(file_path)

#display the first few rows 
print(df.head())

#Basic understanding of data
print("Dataset Info:")
df.info()
print(f"\nSummary statistics:\n{df.describe()}")

#Handling of missing values
missing_values = df.isnull().sum()
print(f"missing value in each column is:")
print(missing_values)

#fill the missing values with median of the respective column
df.fillna(df.median(), inplace= True)
print("\nMissing Values After Treatment:")
print(df.isnull().sum())

#Handling outliers
plt.figure(figsize=(15, 10))
df.boxplot()
plt.xticks(rotation=90)
plt.title('Boxplot of all columns to detect outliers')
plt.show()

#Remove all outliers using Z-score
z_score = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
df_clean = df[(z_score<3).all(axis=1)]
print(f"\nNumber of rows after outlier removal: {df_clean.shape[0]} (original: {df.shape[0]})")

#Analysing variables driirng house prices
corr_matrix = df_clean.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Variables strongly affecting housing rates
important_variables = df_clean.corr()['MEDV'].sort_values(ascending=False)
print("\nVariables most strongly affecting house prices:")
print(important_variables)

#prepare the dataset for linear regression
x = df_clean.drop(columns=['MEDV']) #features
y = df_clean['MEDV'] #Targets

X_train,X_test,y_train,y_test =train_test_split(x,y,test_size = 0.2,random_state = 42)

#initialising and traing the regression model
model = LinearRegression()
model.fit(X_train,y_train)

# Predict on the test set
y_pred = model.predict(X_test)

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")

# Plotting predicted vs actual values
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred, alpha=0.7,edgecolors='k')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max(),],'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

model_path = 'linear_regression_model.pkl'
joblib.dump(model,model_path)
print(f"model saved in {model_path}")