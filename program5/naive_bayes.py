# -*- coding: utf-8 -*-
"""naive_bayes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sxngvOQAAtBhTNZcudwRN5RYSmHgAVdZ

Naive Bayes Classifier using sklearn which includes preprocessing on the titanic dataset
"""

#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.impute import SimpleImputer

#load dataset
df = pd.read_csv('titanic.csv')
df = df[['Survived','Pclass','Age','SibSp','Parch','Fare','Embarked']]

#preprocessing
print(df)

df.describe()

print(df.head())

print(df.info())

print(df.dtypes)

print(df.shape)

#handle missing values
imputer = SimpleImputer(strategy = 'median')
df[['Age','Fare']] = imputer.fit_transform(df[['Age','Fare']])

#fill Embarked values with most frequest values
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace = True)

#encode embarked coloumn
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

#normalize using Standard Scaler or MinMax Scaler
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

#plot the correlation Matrix
plt.figure(figsize = (12,10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix,annot = True,cmap = "coolwarm")
plt.title("correlation matrix")
plt.show()

X = df.drop('Survived',axis = 1)
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(cm)
print(accuracy)

plt.figure(figsize = (12,12))
sns.heatmap(cm,annot = True)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()