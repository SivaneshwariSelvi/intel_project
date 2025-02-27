from sklearn.cluster import KMeans
import numpy as np
data = np.array([[2, 3], [2, 6], [8, 3], [8, 6], [5, 4], [7, 5]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(data)
print("Labels:", kmeans.labels_)
