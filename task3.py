# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:55:41 2020

@author: aashu
"""

import pandas as pd
dataset = pd.read_csv("Iris.csv")

X=dataset.iloc[:, 1:5].values

#preprocessing
#imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer.fit(X[ : ,0:4])

#no need for label encoder and one hot encoder in this case

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)



#k-means clustering
#finding optimum number of clusters from kmeans classification
from sklearn.cluster import KMeans
wcss = []
k=1
totalelements= len(X)


while k < 12:
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
    k=k+1
    
"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
""" 
import matplotlib.pyplot as plt
plt.plot(range(1, 12), wcss)
plt.title('Selecting k with the elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# pick the fewest number of clusters that reduces the average distance
i=0
j=1
diff=[]

while j < len(wcss):
    diff.append(wcss[i]-wcss[j])
    i+=1
    j+=1


threshold = 25
optimalno = 0

while diff[optimalno] >= threshold:
    optimalno=optimalno+1
    
    
print("Optimal no of group is ",optimalno)



# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = optimalno, init = 'k-means++',max_iter = 300,
                n_init = 10, random_state = 0)
ypred = kmeans.fit_predict(X)

# Visualising the clusters - On the first two columns
plt.scatter(X[ypred == 0, 0], X[ypred == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[ypred == 1, 0], X[ypred == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[ypred == 2, 0], X[ypred == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids') 

plt.legend()




#hierarchical clustering

iris_SP= X 

#calculate full dendogram
from scipy.cluster.hierarchy import dendrogram, linkage

#generate the linkage matrix
Z = linkage(iris_SP, 'ward')

#set cutoff ot 150
max_d = 9.08        #max+d as in max distance
plt.figure(figsize = (25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=4.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()



























































