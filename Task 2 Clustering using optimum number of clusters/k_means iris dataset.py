import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from joblib import dump,load

iris_data = datasets.load_iris()
dataset = pd.DataFrame(data = iris_data.data,columns = iris_data.feature_names)
dataset['target'] = iris_data.target

print(dataset.info()) 
print(dataset.describe())
print(dataset.columns)

# histograms or distribution plots
sns.histplot(dataset.loc[:,"sepal length (cm)"])
plt.figure(clear =True)
sns.histplot(dataset.loc[:,"sepal width (cm)"])
plt.figure(clear =True)
sns.histplot(dataset.loc[:,"petal length (cm)"])
plt.figure(clear =True)
sns.histplot(dataset.loc[:,"petal width (cm)"])
plt.figure(clear =True)
#pairplot
sns.pairplot(dataset,hue ="target" )
#barplot
plt.figure(clear =True)
plt.figure(figsize=(10,6))
dataset.iloc[:,:-1].boxplot()


# dividing data into independent and dependent features
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# datapreprocessing (no need to do it data in good)
"""Finding Optimal Number of Clusters for KMeans"""
from sklearn.cluster import KMeans
import os
if not os.path.isfile('kmeans_iris.joblib'):
    wcss = []
    for optimal_clusters in range(1,11):
        kmeans = KMeans(n_clusters = optimal_clusters, 
                        init='k-means++',
                        n_init=10, 
                        max_iter=300,
                        random_state = 42)
        kmeans.fit(X)
        loss = kmeans.inertia_
        wcss.append(loss)
    #plotting loss vs number of clusters (elbow method)
    plt.plot(range(1,11),wcss,'r')
    plt.xlabel('no. of clusters')
    plt.ylabel('loss or wcss')
    
    # optimal numbers of clusters = 3
    kmeans = KMeans(n_clusters = 3, 
                        init='k-means++',
                        n_init=10, 
                        max_iter=300,
                        random_state = 42)
    kmeans.fit(X)
    
    dump(kmeans,'kmeans_iris.joblib')
else:
    kmeans = load('kmeans_iris.joblib')
# retrieving saved model


clustered_points = kmeans.predict(X)
# visualizing clusters
plt.figure(clear = True)
plt.scatter(x = X[clustered_points == 0,2],y = X[clustered_points == 0,3],c = 'red',label = 'setosa')
plt.scatter(x = X[clustered_points == 1,2],y = X[clustered_points ==1 , 3],c = 'blue',label = 'versi color')
plt.scatter(x = X[clustered_points == 2,2],y = X[clustered_points ==2 , 3],c = 'green',label = 'virginica')
plt.scatter(x = kmeans.cluster_centers_[:,2],y = kmeans.cluster_centers_[:,3],c = 'yellow',label = 'centroids')
plt.legend()
plt.show()


    
    

