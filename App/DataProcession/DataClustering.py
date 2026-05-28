# if __name__ == '__main__':

"""

    Class for data clustering process (Кластеризація)
    https://builtin.com/data-science/data-clustering-python

    !!!!!!!!!!!!!!!!
    https://realpython.com/k-means-clustering-python/

"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from PyQt5.QtWidgets import QApplication

# path = "D:\PythonProjects\MagRobota\App\Interface\Set1.xlsx"
# initialData = pd.read_excel(path, sheet_name=0)
#
# data = []
#
# labels = initialData[["Свій/Чужий"]].copy()
# features = initialData.iloc[:, 2:-1]
#
# Y = []
# X = []
#
# for i in range(len(labels)):
#     subArr = []
#     for j in range(len(features.iloc[i, :])):
#         subArr.append(features.iloc[i, :][j])
#
#     Y.append(labels.iloc[i, :][0])
#     X.append(subArr)


def getData(x):

    global X
    # global Y

    X = x
    # Y = y

    print("Features==============================")
    print(X)
    # print("Labels================================")
    # print(Y)

def getClustersNumber():

    # Scale features
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(X)

    # Get number of clusters
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42, }

    # A list holds the SSE values for each k
    sse = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaledFeatures)
        sse.append(kmeans.inertia_)

    # plt.style.use("fivethirtyeight")
    # plt.plot(range(1, 11), sse)
    # plt.xticks(range(1, 11))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.show()

    # Knee locator
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print(kl.elbow)

    # The silhouette coefficient is a measure of cluster cohesion and separation.
    # It quantifies how well a data point fits into its assigned cluster based on two factors:
    #
    # How close the data point is to other points in the cluster
    # How far away the data point is from points in other clusters

    # Silhouette coefficient
    silhouetteCoefficients = []

    # Notice you start at 2 clusters for silhouette coefficient

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaledFeatures)
        score = silhouette_score(scaledFeatures, kmeans.labels_)
        silhouetteCoefficients.append(score)

    # plt.style.use("fivethirtyeight")
    # plt.plot(range(2, 11), silhouetteCoefficients)
    # plt.xticks(range(2, 11))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Silhouette Coefficient")
    # plt.show()

    return sse, silhouetteCoefficients

def getClusters(n):

    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(X)

    # print(scaledFeatures)

    kmeans = KMeans(init="random", n_clusters=n, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(scaledFeatures)

    # The lowest SSE value
    print(kmeans.inertia_)
    # Final locations of the centroid
    print(kmeans.cluster_centers_)
    # The number of iterations required to converge
    print(kmeans.n_iter_)
    # Predicted lables
    clustered = kmeans.labels_
    print(clustered)

    cY = clustered
    cX = []
    for i in range(len(X)):
        cX.append(sum(X[i]) / len(X[i]))

    # classColormap = matplotlib.colors.ListedColormap(['#FF0000', '#00FF00', '#000000'])
    # plt.scatter(cY, cX, c=cY, cmap=classColormap)
    # plt.scatter(range(len(cX)), cX, c=cY, cmap=classColormap)

    # for k in range(0, len(X)):
    #     x = sum(X[k]) / len(X[k])
    #     plt.scatter(x, k)

    # for k in range(0, len(X)):
    #     x = sum(X[k]) / len(X[k])
    #     plt.scatter(Y[k], x)

    # plt.style.use("fivethirtyeight")
    # plt.title("Clusters Identified by K-Means")
    # plt.ylabel("Середнє значення Амп")
    # plt.xlabel("Групи")
    # plt.show()

    return cY, cX





# def getData(initialData):
#
#     X = initialData[['Свій/Чужий', '16 Гц Амп']].copy()
#     print(X)
#
#     return X
#
# def getNumberOfClusters(data):
#
#     # Take data y, and x (our identifiers, and all data for our process)
#     print(data)
#
#     wcss = []
#     for i in range(1, 11):
#         kmeans = KMeans(n_clusters=i, random_state=0)
#         kmeans.fit(data)
#         wcss.append(kmeans.inertia_)
#
#     sns.set()
#     plt.plot(range(1, 11), wcss)
#
#     plt.title('Selecting the Numbeer of Clusters using the Elbow Method')
#     plt.xlabel('Clusters')
#     plt.ylabel('WCSS')
#     plt.show()
#
# def getClusters(numberOfClusters, clusteringData):
#
#     from sklearn.mixture import GaussianMixture
#
#     n_clusters = numberOfClusters
#     gmm_model = GaussianMixture(n_components=n_clusters)
#     gmm_model.fit(clusteringData)
#
#     cluster_labels = gmm_model.predict(clusteringData)
#     X = pd.DataFrame(clusteringData)
#     X['cluster'] = cluster_labels
#
#     for k in range(0, n_clusters):
#         data = X[X["cluster"] == k]
#         plt.scatter(data['Свій/Чужий'], data['16 Гц Амп'])
#
#     plt.title("Clusters Identified by Guassian Mixture Model")
#     plt.ylabel("16 Гц Амп")
#     plt.xlabel("Свій/Чужий")
#     plt.show()
#
# X = getData(initialData)
#
# getNumberOfClusters(X)
#
# getClusters(2, X)




