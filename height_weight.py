import pandas as pd

df = pd.read_csv('height_weight_data.csv')
df

df.shape

df.describe()

x = df.iloc[:,[0,1]].values
x

x.ndim

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 2)
model.fit(x)

y_pred = model.predict(x)
y_pred

x[y_pred == 0,0] #cluster 0,Column 0

x[y_pred == 0,1] #cluster 0, Column 1

x[y_pred == 1,0] #cluster 1, Column 0

x[y_pred == 1,1] #cluster 1,Column 1

import matplotlib.pyplot as plt
plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1], c = 'r', s = 50, label = 'Cluster 1')
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], c = 'g', s = 50, label = 'Cluster 2')
centroids = model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='b', s=200, marker='*', label='Centroids')

# Add labels and legend
plt.title('Clusters and Centroids')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()
