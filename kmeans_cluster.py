import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

# Sun'iy ma'lumotlarni yaratish
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.60)

# Sun'iy ma'lumotlarni tasvirlash
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Sun'iy ma'lumotlar")
plt.show()

# K-Means klasterlash
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# K-Means natijalarini tasvirlash
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
plt.title("K-Means Klasterlash")
plt.show()

# Elbow Method yordamida optimal klasterlar sonini aniqlash
inertia_values = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia_values, 'bo-')
plt.xlabel('Klasterlar soni')
plt.ylabel('Inertia')
plt.title("Elbow Method orqali optimal klasterlar sonini aniqlash")
plt.show()

# DBSCAN klasterlash
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# DBSCAN natijalarini tasvirlash
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='plasma')
plt.title("DBSCAN Klasterlash")
plt.show()
