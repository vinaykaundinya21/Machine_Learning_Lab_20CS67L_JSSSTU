#!/usr/bin/env python
# coding: utf-8

# In[14]:


#6 -> KNN on glass dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer


# In[15]:


df = pd.read_csv('glass.csv')
print(df)


# In[16]:


df.head()


# In[17]:


df.info()


# In[18]:


df.dtypes


# In[19]:


df.shape


# In[20]:


print(df.isnull().sum())


# In[21]:


df.describe()


# In[27]:


#handle missing values but not present
imputer = SimpleImputer(strategy = 'median')
df[df.columns] = imputer.fit_transform(df[df.columns])


# In[28]:


#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Type',axis = 1))


# In[30]:


#plot correlation Matrix
plt.figure(figsize = (12,10))
correlation = df.corr()
sns.heatmap(correlation,annot = True,cmap = "coolwarm")
plt.title('correlation matirx')
plt.show()


# In[32]:


#assign features target 
X = X_scaled
y = df['Type'].values
print(X)
print(y)


# In[33]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[34]:


#defining a manhattan and euclidean dist
def custom_euclidean(x1,x2):
    return np.sqrt(np.sum((x1-x2) ** 2))
def custom_manhattan(x1,x2):
    return np.sum(np.abs(x1-x2))


# In[36]:


#intialize knn classifier
k = 3
clf_custom_euclidean = KNeighborsClassifier(n_neighbors = k,metric = custom_euclidean)
clf_custom_manhattan = KNeighborsClassifier(n_neighbors = k,metric = custom_manhattan)


# In[37]:


#fitt knn
clf_custom_euclidean.fit(X_train,y_train)
clf_custom_manhattan.fit(X_train,y_train)


# In[38]:


#make predictions
prediction_custom_euclidean = clf_custom_euclidean.predict(X_test)
prediction_custom_manhattan = clf_custom_manhattan.predict(X_test)


# In[39]:


#calculate accuracy
accuracy_custom_euclidean = accuracy_score(y_test,prediction_custom_euclidean)
accuracy_custom_manhattan = accuracy_score(y_test,prediction_custom_manhattan)
print("accuracy_custom_euclidean : ",accuracy_custom_euclidean)
print("accuracy_custom_manhattan : ",accuracy_custom_manhattan)


# In[40]:


confusion_custom_euclidean = confusion_matrix(y_test,prediction_custom_euclidean)
confusion_custom_manhattan = confusion_matrix(y_test,prediction_custom_manhattan)
print("confusion_custom_euclidean: ",confusion_custom_euclidean)
print("confusion_custom_manhattan : ",confusion_custom_manhattan)


# In[41]:


plt.figure(figsize = (10,10))
sns.heatmap(confusion_custom_euclidean,annot = True,cmap = 'Blues')
plt.title('Confusion matrix heatmap')
plt.show()


# In[42]:


plt.figure(figsize = (10,10))
sns.heatmap(confusion_custom_manhattan,annot = True,cmap = 'Blues')
plt.title('Confusion matrix heatmap')
plt.show()


# In[ ]:




