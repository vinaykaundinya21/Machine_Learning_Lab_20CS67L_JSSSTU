#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Kmeans -> unsupervised
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns


# In[19]:




def kmeans(X, K, max_iters=100):
    # Step 1: Initialize centroids with the first K samples
    centroids = X[:K]
    
    # Step 1: Assign the remaining n-K samples to the nearest centroid and update centroids
    for i in range(K, len(X)):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(distances)
        centroids[nearest_centroid] = (centroids[nearest_centroid] + X[i]) / 2
    
    labels = np.zeros(X.shape[0])
    
    # Step 2: Assign each sample to the nearest centroid without updating centroids
    for _ in range(max_iters):
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            nearest_centroid = np.argmin(distances)
            labels[i] = nearest_centroid
    
    return labels, centroids


# In[20]:


iris = load_iris()
X = iris.data
y = iris.target
print(X)
print(y)


# In[21]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[22]:


correlation_matrix = np.corrcoef(X_scaled.T)
plt.figure(figsize = (12,12))
sns.heatmap(correlation_matrix,annot = True,cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[23]:


K = 3
labels,centroids = kmeans(X_scaled,K)
print("labels : ",labels)
print("centroids",centroids)


# In[26]:


plt.scatter(X_scaled[:,0],X_scaled[:,1],c = labels)
plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',color = 'red',s = 200)
plt.xlabel('Sepal lenght')
plt.ylabel('sepal width')
plt.title('K means of iris')
plt.show()


# In[27]:


conf_matrix = confusion_matrix(y,labels)
plt.figure(figsize=(12,12))
sns.heatmap(conf_matrix,annot = True,cmap = "Blues")
plt.xlabel("predicted")
plt.ylabel('True')
plt.title("confusion matirx heatmao")
plt.show()


# In[ ]:




