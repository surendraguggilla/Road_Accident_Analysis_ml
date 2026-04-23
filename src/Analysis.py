

## ===========  IMPORTING LIBRARIES  ===================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## ============  LOAD DATA  ===========================
print("")
df = pd.read_excel("D:\\Road_Accident_analysis\\Data\\road_accidents_states_wise.xlsx")
print("DATASET LOADED SUCCESSFULLY")
print("HEAD: \n", df.head(), "\n")

## ===========  DATA EXPLORATION  ======================

print(df.info(), "\n")
print("Missing Values: \n",df.isnull().sum())
print("Duplicated Values: ",df.duplicated().sum())

data = df[['Total_Accidents', 'Deaths', 'Injuries']]
data = data.fillna(data.mean())

## =========== BASIC ANALYSIS =======================
print("Summary:")
print(data.describe(), "\n")


## ========== VISUALIZATION ==========================

# Bar Chart
df.groupby('State')['Total_Accidents'].sum().head(10).plot(kind='bar', color='blue')
plt.title("Top States Accidents")
plt.xticks(rotation=45)
plt.show()

# Line Chart
df.groupby('Year')['Total_Accidents'].sum().plot(color='green')
plt.title("Accidents by Year")
plt.show()

# Pie Chart
df['Road_Type'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
plt.title("Road Type")
plt.ylabel("")
plt.show()

# Histogram
plt.figure()
plt.hist(data['Total_Accidents'], color='orange')
plt.title("Accident Distribution")
plt.show()

# Box Plot
plt.figure()
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()

# Scatter Plot
plt.figure()
sns.scatterplot(x=data['Total_Accidents'], y=data['Deaths'], color='red')
plt.title("Accidents vs Deaths")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Count Plot
plt.figure()
sns.countplot(x='Road_Type', data=df)
plt.xticks(rotation=45)
plt.title("Road Type Count")
plt.show()

# Area Plot
plt.figure()
df.groupby('Year')['Total_Accidents'].sum().plot(kind='area', alpha=0.4)
plt.title("Accident Trend")
plt.show()

## ==============  HYPOTHESIS TESTING  =====================

# T-Test
g1 = df[df['Year'] <= 2019]['Total_Accidents']
g2 = df[df['Year'] > 2019]['Total_Accidents']

t, p = ttest_ind(g1, g2)

print("\nT-Test Results:")
print("p-value:", p)

if p < 0.05:
    print("There is a significant difference in accidents before and after 2019.")
else:
    print("No significant difference found.")

# Pearson Correlation Test
c, p2 = pearsonr(data['Total_Accidents'], data['Deaths'])
 
print("\nCorrelation Test Results:")
print("Correlation:", c)
print("p-value:", p2)
 
## =============  MODEL TRAINING  ====================
 
## ============  SIMPLE LINEAR REGRESSION  =============
 
X = data[['Deaths', 'Injuries']]
y = data['Total_Accidents']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
print("\n====== Linear Regression Results ======")
print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Coefficients   : Deaths = {model.coef_[0]:.4f}, Injuries = {model.coef_[1]:.4f}")
 
## ========== MODEL EVALUATION ===================
 
pred = model.predict(X_test)
 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
mae  = mean_absolute_error(y_test, pred)
mse  = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, pred)
 
print("\n====== Model Evaluation Metrics ======")
print(f"Mean Absolute Error  (MAE) : {mae:.4f}")
print(f"Mean Squared Error   (MSE) : {mse:.4f}")
print(f"Root Mean Sq. Error (RMSE) : {rmse:.4f}")
print(f"R² Score                   : {r2:.4f}")
 
## ========== ACTUAL VS PREDICTED PLOT ===================
 
sns.set_style("whitegrid")
 
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=pred, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
