import numpy as np
import cv2
import math
import copy
import skimage.segmentation as ss


class SuperPixel:
    def __init__(self, centerLoc, centerLab):
        self.centerLoc = centerLoc
        self.centerLab = centerLab

    def moveCenterTo(self, img, i, j):
        self.centerLoc = [i, j]
        self.centerLab = img[self.centerLoc[0], self.centerLoc[1], :]


# def drawBoundary(img, labels):
#     t = 2
#     hei, wid = labels.shape
#     a = labels[0, 0]
#     for i in range(hei):
#         for j in range(wid):
#             b = labels[i, j]
#             if (a != b):
#                 img[(i - t):(i + t), (j - t):(j + t), :] = [0, 0, 0]
#             a = b
#     a = labels[0, 0]
#     for j in range(wid):
#         for i in range(hei):
#             b = labels[i, j]
#             if (a != b):
#                 img[(i - t):(i + t), (j - t):(j + t), :] = [0, 0, 0]
#             a = b
#     return img

def calError(superPixels1, superPixels2):
    error = 0
    for s in range(len(superPixels1)):
        error = error + math.sqrt(
            np.sum((np.array(superPixels1[s].centerLoc) - np.array(superPixels2[s].centerLoc)) ** 2))
    error = error / len(superPixels1)
    return error


def getGradient(img, i, j):
    gradient = np.sum((img[i + 1, j, :] - img[i - 1, j, :]) ** 2) + np.sum((img[i, j + 1, :] - img[i, j - 1, :]) ** 2)
    return gradient


def moveCenters(img, superPixels, windowSize=5):
    a = int((windowSize - 1) / 2)
    for s in range(len(superPixels)):
        centerLoc = superPixels[s].centerLoc
        gradientMin = float('inf')
        besti, bestj = 0, 0
        for i in range(centerLoc[0] - a, centerLoc[0] + a + 1):
            for j in range(centerLoc[1] - a, centerLoc[1] + a + 1):
                try:
                    gradient = getGradient(img, i, j)
                    if (gradient < gradientMin):
                        gradientMin = gradient
                        besti, bestj = i, j
                except:
                    continue
        superPixels[s].moveCenterTo(img, besti, bestj)
    return superPixels


def initiate(img, K):
    height, width, channels = img.shape
    Si = height / math.sqrt(K)
    Sj = width / math.sqrt(K)
    S = int((Si + Sj) / 2)
    superPixels = []
    for i in range(int(Si / 2), height, int(Si)):
        for j in range(int(Sj / 2), width, int(Sj)):
            superPixels.append(SuperPixel([int(i), int(j)], img[int(i), int(j), :]))
    return S, superPixels


def iterate(img, indices, superPixels, S, alpha):
    height, width, channels = img.shape
    labels = np.full([height, width], -1).astype(int)
    minDists = np.full([height, width], float('inf'), dtype='uint64')
    Len = len(superPixels)
    for s in range(Len):
        print("\r", str(int((s + 1) / Len * 100)), "%", end="")
        centerLoc, centerLab = superPixels[s].centerLoc, superPixels[s].centerLab
        iSlice = slice((max((centerLoc[0] - S), 0)), (min((centerLoc[0] + S + 1), height)))
        jSlice = slice((max((centerLoc[1] - S), 0)), (min((centerLoc[1] + S + 1), width)))
        neighborLab = img[iSlice, jSlice, :]
        neighborLoc = indices[iSlice, jSlice, :]
        distLab = np.sum((neighborLab - centerLab) ** 2, axis=2)
        distLoc = np.sum((neighborLoc - centerLoc) ** 2, axis=2)
        dist = distLab + alpha * distLoc
        i, j = np.where(dist < minDists[iSlice, jSlice])
        (minDists[iSlice, jSlice])[i, j] = dist[i, j]
        (labels[iSlice, jSlice])[i, j] = s

    for s in range(len(superPixels)):
        centerLoc, centerLab = superPixels[s].centerLoc, superPixels[s].centerLab
        iSlice = slice((max((centerLoc[0] - S), 0)), (min((centerLoc[0] + S + 1), height)))
        jSlice = slice((max((centerLoc[1] - S), 0)), (min((centerLoc[1] + S + 1), width)))
        i, j = np.where(labels[iSlice, jSlice] == s)
        mean = np.mean(indices[iSlice, jSlice][i, j, :], axis=0).astype(int)
        superPixels[s].moveCenterTo(img, mean[0], mean[1])
    return labels, superPixels


def smooth(superPixels, labels, S):
    k = int(S / 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    height, width = labels.shape
    print("smoothing : ")
    Len = len(np.unique(labels))
    for s in range(Len):
        print("\r", str(int((s + 1) / Len * 100)), "%", end="")
        centerLoc, centerLab = superPixels[s].centerLoc, superPixels[s].centerLab
        iSlice = slice((max((centerLoc[0] - S), 0)), (min((centerLoc[0] + S + 1), height)))
        jSlice = slice((max((centerLoc[1] - S), 0)), (min((centerLoc[1] + S + 1), width)))
        labelsCropped = labels[iSlice, jSlice]
        mask = np.zeros(labelsCropped.shape)
        i, j = np.where(labelsCropped == s)
        mask[i, j] = 1
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        i, j = np.where(mask == 1)
        labelsCropped[i, j] = s
        labels[iSlice, jSlice] = labelsCropped
    return labels

def slic(img, K, alpha):
    imgSlic = img.copy()
    height, width, channels = imgSlic.shape
    imgSlic = cv2.cvtColor(imgSlic, cv2.COLOR_BGR2LAB)
    indices = np.indices((height, width), dtype=int)
    indices = np.dstack((indices[0], indices[1]))
    S, superPixels = initiate(imgSlic, K)
    print("\nK =", K, "   S =", S)
    superPixels = moveCenters(imgSlic, superPixels, 5)
    superPixelsOld = copy.deepcopy(superPixels)
    limit = 1.5
    for i in range(20):
        print("iteration", str(i + 1), ": ")
        labels, superPixels = iterate(imgSlic, indices, superPixels, S, alpha)
        error = calError(superPixelsOld, superPixels)
        print('   Error:', str(error))
        if (error < limit):
            break
        superPixelsOld = copy.deepcopy(superPixels)
    labels = smooth(superPixels, labels, S)
    imgSlic = ss.mark_boundaries(image=img, label_img=labels, color=(0, 0, 0), mode='thick')
    imgSlic = cv2.normalize(imgSlic, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(int)
    return imgSlic


I = cv2.imread("slic.jpg")
cv2.imwrite("res05.jpg", slic(I, K=64, alpha=0.005))
I = cv2.imread("slic.jpg")
cv2.imwrite("res06.jpg", slic(I, K=256, alpha=0.01))
I = cv2.imread("slic.jpg")
cv2.imwrite("res07.jpg", slic(I, K=1024, alpha=0.05))
I = cv2.imread("slic.jpg")
cv2.imwrite("res08.jpg", slic(I, K=2048, alpha=0.1))
