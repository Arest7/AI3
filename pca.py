import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Sun'iy ma'lumotlarni 3 o'lchamda yaratish
X, y = make_blobs(n_samples=300, centers=3, n_features=3, random_state=42, cluster_std=1.0)

# PCA yordamida 3D ma'lumotlarni 2D ga qisqartirish
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means klasterlash
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_pca)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# PCA yordamida qisqartirilgan va K-Means yordamida klasterlash qilingan ma'lumotlarni tasvirlash
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
plt.title("PCA bilan o'lchamlarni qisqartirish va K-Means klasterlash")
plt.xlabel("Asosiy komponent 1")
plt.ylabel("Asosiy komponent 2")
plt.show()
