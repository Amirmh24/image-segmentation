from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

file = open("Points.txt", 'r')
n = int(file.readline())
points = np.zeros((n, 2))
for i in range(n):
    line = file.readline()
    x = float(line.split(' ')[0])
    y = float(line.split(' ')[1])
    points[i] = [x, y]

plt.plot(points[:, 0], points[:, 1], 'o', markersize=4, color='black')
plt.savefig("res01.jpg")
plt.clf()

kmeans = KMeans(n_clusters=2).fit(points)
colors = np.array(['black', 'red'])
plt.scatter(points[:, 0], points[:, 1], s=16, c=colors[kmeans.labels_])
plt.savefig("res02.jpg")
plt.clf()

dists = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
dists = dists.reshape([-1, 1])
kmeans = KMeans(n_clusters=2).fit(dists)
plt.scatter(points[:, 0], points[:, 1], s=16, c=colors[kmeans.labels_])
plt.savefig("res03.jpg")
