# -*- coding: utf-8 -*-
"""DescionTree.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hoQ_DNoAO1sMml7YASikHSL3LlL_Z78O

Descion Tree Classifier - > weather forecasting dataset using sklearn
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#laod dataset
df = pd.read_csv('weather_forecast.csv')

df.head()

print(df)

print(df.info())

df.describe()

df.dtypes

df.shape

#preprocessing : convert catergorical to numerical
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(df.drop('Play',axis = 1)).toarray()
print(X_encoded)

y = df['Play']

X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,test_size = 0.2,random_state = 42)

clf_id3 = DecisionTreeClassifier(criterion='entropy',random_state = 42)
clf_id3.fit(X_train,y_train)

plt.figure(figsize = (12,10))
plot_tree(clf_id3, filled = True,feature_names=encoder.get_feature_names_out(['Outlook','Temperature','Humidity','Windy']),class_names=['No','Yes'])
plt.show()

y_pred_id3 = clf_id3.predict(X_test)

accuracy_id3 = accuracy_score(y_test,y_pred_id3)
print(accuracy_id3)
print(classification_report(y_test,y_pred_id3))
print(classification_report)