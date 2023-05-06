# I Have created the multivariate Linear Regression model on the Computer Hardware Dataset
# To Predict the relative CPU performance of computer 

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Reading The Dataset
df = pd.read_csv('machine.csv')

# Data Preprocessing

print(df.info())
print(df.head())
print(df.describe())

# PairPlotting
sns.pairplot(df,x_vars=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX'], y_vars=['ERP'])
plt.show()

# Scatter Plotting
plt.scatter(df['MYCT'], df['ERP'])
plt.xlabel('MYCT')
plt.ylabel('Relative CPU Performance')
plt.show()


#Correlation Matrix
corr = df[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX' ,'ERP']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


# Splitting The Dataset into Testing and Training Data

# Multivariate Linear Regression Model
features = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']

X_train, X_test, y_train, y_test = train_test_split(df[features].values, df['PRP'].values, test_size=0.2, random_state=42)


# Fitting into the Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# Evaluating the Linear Regression Model

# Make predictions on the test data
y_pred = model.predict(X_test)
print(y_pred)

r2 = r2_score(y_test, y_pred)
r_m_square = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R-squared: {r2:.2f}")
print(f"Root mean squared error: {r_m_square:.2f}")