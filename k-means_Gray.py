import numpy as np
import pylab as plt
import PIL.Image as image
import matplotlib.pyplot as plt1
from sklearn.cluster import KMeans

img = plt.imread('oct.jpeg')
img1 = img.reshape((img.shape[0]*img.shape[1], 1))
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(img1)
height = img.shape[0] 
width = img.shape[1]
pic_new = image.new("L", (width, height))
center = np.zeros([k, 1])

for i in range(k):
    for j in range(1):
        center[i, j] = kmeans.cluster_centers_[i, j]
center = center.astype(np.int32)
label = kmeans.labels_.reshape((height, width))
for i in range(height):
    for j in range(width):
        pic_new.putpixel((j, i), int(center[label[i][j]]))
pic_new.save("Gray_out.jpg", "JPEG")