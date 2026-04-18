
## ===========  IMPORTING LIBRARIES  ===================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

## ============  LOAD DATA  ===========================

df = pd.read_csv("D:/crop_recommendation/Data/crop_recommendation.csv")
print("DATASET LOADED SUCCESSFULLY")
print("HEAD: \n",df.head(),"\n")

## ===========  DATA EXPLORATION  ======================

print(df.info(),"\n")
print(df.describe(),"\n")
print("\nInitial Shape: ",df.shape)
print("\nMissing Values: \n",df.isnull().sum())
print("\nCOLUMNS: \n",df.columns,"\n")

## ========== DISTRIBUTION OF DATA ===================

plt.figure()
sns.countplot(x="label", data=df)
plt.xticks(rotation=90) #to prevent overlapping of x-axis
plt.title("Crop Distribution")
plt.show()

## ========== VISUALIZATION ==========================

num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

## =========== OUTLIAR DETECTION AND HANDLING ==============

#finding outliars
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Handling outliars
#for right-skewed
df["P"] = np.log1p(df["P"])
df["K"] = np.log1p(df["K"])

#for left-skewed
df["humidity"] = np.sqrt(df["humidity"])
df["rainfall"] = np.sqrt(df["rainfall"])


## ==============  CORRELATION HEAT-MAP  =====================

#confirm only numeric columns are selected
df_num = df.select_dtypes(include = 'number')
plt.figure()
sns.heatmap(df_num.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

#To know the relationship with target
plt.figure()
sns.boxplot(x='label',y='rainfall',data=df)
plt.title("Rainfall Vs Target")
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.boxplot(x='label',y='N',data=df)
plt.title("Nitrogen vs Target")
plt.xticks(rotation=90)
plt.show()


## =============  MODEL-TRAINING  ====================


## ============ RANDOM FOREST CLASSIFIER  =============

x = df.drop('label',axis=1)
y = df['label']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n[+] Model saved as 'model.pkl' successfully!")

rf_pred = rf_model.predict(x_test)
print("Random Forest Accuracy: ",accuracy_score(y_test,rf_pred))
print("\nRandom Forest Classification Report:\n",classification_report(y_test,rf_pred))


## ===========  LOGISTIC REGRESSION  ================


# ====== FEATURE SCALING ===========
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


lr_model = LogisticRegression(max_iter=2200)
lr_model.fit(x_train_scaled, y_train)

lr_pred = lr_model.predict(x_test_scaled)
print("\nLogistic Regression Accuracy: ",accuracy_score(y_test,lr_pred))
print("\nLogistic Regression Classification Report:\n",classification_report(y_test,lr_pred))


## ======= IMPORTANCE OF FEATURES ON MODEL  ==============

importances = rf_model.feature_importances_
feature_names = x.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
    }).sort_values(by='Importance',ascending=False)
print("\nFeature Importance:\n",feat_df)


## ========== COMPARING MODEL ACCURACIES =================


rf_acc = accuracy_score(y_test,rf_pred)
lr_acc = accuracy_score(y_test,lr_pred)

print("\n Model Comparison: ")
print("Random Forest Accuracy: ",rf_acc)
print("Logistic Regression Accuracy: ",lr_acc,"\n")

if(lr_acc >rf_acc):
   print("Logistic Regression performs better.\n")
else:
    print("Random Forest performs better.\n")


## ========== CONFUSION MATRIX =================

cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm,annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.show()


## ========= PREDICTION FOR USER-GIVEN VALUES ================

print("\nEnter values for prediction: ")

N = float(input("Nitrogen(N): "))
P = float(input("Phosphorus(P): "))
K = float(input("Potassium(K): "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

#Transforming data
P = np.log1p(P)
K = np.log1p(K)
humidity = np.sqrt(humidity)
rainfall = np.sqrt(rainfall)

user_df = pd.DataFrame([{
    'N': N,
    'P': P,
    'K': K,
    'temperature': temperature,
    'humidity': humidity,
    'ph': ph,
    'rainfall': rainfall
}])

prediction = rf_model.predict(user_df)

print("\n Best Recommended Crop: ",prediction[0])




