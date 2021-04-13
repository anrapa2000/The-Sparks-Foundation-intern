#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Internship - The Sparks Foundation

# ## Author - C Aparna

# ### Task 2:
# #### Problem : From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually. 

# ##### Importing libraries

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# ##### Loading the Iris Dataset

# In[4]:


iris = datasets.load_iris() 
#iris dataset available in the library package sklearn
data_df = pd.DataFrame(iris.data, columns = iris.feature_names)
data_df.head()
#returns the top n rows,here top 5 rows by default


# In[5]:


data_df.tail()
#return the last n rows, here last 5 rows by default


# In[6]:


data_df.describe()
#returns the summary of basic stastical details


# In[7]:


data_df.shape
#returns the number of rows,columns in the dataset


# In[8]:


data_df.isnull().sum()
#to check if any of the values in the dataset is null


# In[9]:


data_df.info()


# ##### Finding the optimal number of clusters for K-Means

# In[11]:


from sklearn.cluster import KMeans


# In[12]:


x = data_df.iloc[:, [0, 1, 2, 3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
wcss #within cluster sum of squares


# In[13]:


#Plotting the results onto a line graph
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# In[14]:


#Applying kmeans to the dataset 
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[15]:


#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# ### Thank you! 
