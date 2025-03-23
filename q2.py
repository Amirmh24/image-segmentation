import numpy as np
import cv2
from skimage.color import label2rgb
import sklearn.cluster as sc

a = 4
I = cv2.imread("park.jpg")
height, width, channels = I.shape
I = cv2.resize(I, (int(width / a), int(height / a)))
height, width, channels = I.shape
reshapedI = np.reshape(I, [-1, 3])
meanShift=sc.MeanShift(bandwidth=10,min_bin_freq=100,bin_seeding=True).fit(reshapedI)
labels=meanShift.labels_
labels=np.reshape(labels, I.shape[:2])
print(np.unique(labels))
kernel = np.ones((3, 3), np.uint8)
for l in range(len(np.unique(labels))):
    mask = np.zeros(labels.shape)
    i,j=np.where(labels==l)
    mask[i,j]=1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    i, j = np.where(mask == 1)
    labels[i,j]=l
I=label2rgb(labels,I,kind='avg')
I = cv2.resize(I, (int(width * a), int(height * a)))
cv2.imwrite("res04.jpg", I)
