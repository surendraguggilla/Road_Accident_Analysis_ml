
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/crop_recommendation/Data/crop_recommendation.csv")
print("DATASET LOADED SUCCESSFULLY")
print("HEAD: \n",df.head())

#Exploring dataset
print(df.info())
print(df.describe())
print("COLUMNS: \n",df.columns)

#Exploring Distribution of Data
plt.figure()
sns.countplot(x="label", data=df)
plt.xticks(rotation=90) #to prevent overlapping of x-axis
plt.title("Crop Distribution")
plt.show()

#visualization for skewness
num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

#finding outliars
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

#Handling outliars
#for right-skewed
df["N"] = np.log1p(df["N"])
df["P"] = np.log1p(df["P"])
df["K"] = np.log1p(df["K"])

#for left-skewed
df["humidity"] = np.sqrt(df["humidity"])
df["rainfall"] = np.sqrt(df["rainfall"])

#Corelation
#confirm only numeric columns are selected
df_num = df.select_dtypes(include = 'number')
plt.figure()
sns.heatmap(df_num.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

#To know the relationship with target
plt.figure()
sns.boxplot(x='label',y='rainfall',data=df)
plt.xticks(rotation=90)
plt.show()

plt.figure()
sns.boxplot(x='label',y='N',data=df)
plt.xticks(rotation=90)
plt.show()

