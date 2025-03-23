import skimage.segmentation as ss
import numpy as np
import cv2


# I = cv2.imread("birds.jpg")
# labels = ss.felzenszwalb(I, scale=200, sigma=4, min_size=100000)
# i, j = np.where(labels != 1)
# I[i, j, :] = 0
# cv2.imwrite("res09-.jpg", I)

def cut(img, x1, y1, x2, y2, w, h):
    hei = x2 - x1
    wid = y2 - y1
    imgGround = img[x1 - h:x2 + h, y1 - w:y2 + w, :]
    rect = (w, h, wid, hei)
    mask = np.zeros(imgGround.shape[:2], np.uint8)
    bgdMode = np.zeros((1, 65), np.float64)
    fgdMode = np.zeros((1, 65), np.float64)
    cv2.grabCut(imgGround, mask, rect, bgdMode, fgdMode, 1, cv2.GC_INIT_WITH_RECT)
    maskNew = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = imgGround * maskNew[:, :, np.newaxis]
    return result[h:h + hei, w:w + wid, :]


def cleanList(imask, jmask):
    imaskCleaned, jfmaskCleaned = np.array([imask[0]]), np.array([jmask[0]])
    for i in range(len(imask)):
        s = True
        k = len(imaskCleaned)
        for j in range(k):
            dist = ((imask[i] - imaskCleaned[j]) ** 2 + (jmask[i] - jfmaskCleaned[j]) ** 2) ** (1 / 2)
            if (dist < 50):
                s = False
                break
        if (s == True):
            imaskCleaned = np.append(imaskCleaned, imask[i])
            jfmaskCleaned = np.append(jfmaskCleaned, jmask[i])
    return imaskCleaned, jfmaskCleaned


def findAll(img, res, temp, alpha, w, h):
    hei, wid, chan = temp.shape
    imgTmp = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    imask, jmask = np.where(imgTmp > alpha)
    imask, jmask = cleanList(imask, jmask)
    for i in range(len(imask)):
        print("\r", str(int(i / (len(imask) - 1) * 100)), "%", end="")
        x1, y1 = imask[i], jmask[i]
        x2, y2 = imask[i] + hei, jmask[i] + wid
        templateCut = cut(I, x1, y1, x2, y2, w, h)
        if (np.sum(templateCut) != 0):
            res[x1:x2, y1:y2, :] = templateCut
    return res


I = cv2.imread("birds.jpg")
res = np.zeros(I.shape)

x1, y1 = 1540, 4000
x2, y2 = 1670, 4120
tem1 = I[x1:x2, y1:y2, :]
res = findAll(I, res, tem1, 0.55, 10, 600)
print()
x1, y1 = 2097, 1229
x2, y2 = 2200, 1300
tem2 = I[x1:x2, y1:y2, :]
res = findAll(I, res, tem2, 0.70, 10, 200)
print()
x1, y1 = 2240, 221
x2, y2 = 2340, 310
tem3 = I[x1:x2, y1:y2, :]
res = findAll(I, res, tem3, 0.73, 10, 600)
print()
x1, y1 = 1920, 2326
x2, y2 = 2024, 2438
tem4 = I[x1:x2, y1:y2, :]
res = findAll(I, res, tem4, 0.7, 10, 600)
print()
x1, y1 = 2090, 1030
x2, y2 = 2220, 1111
tem5 = I[x1:x2, y1:y2, :]
res = findAll(I, res, tem5, 0.8, 10, 800)

i, j, k = np.where(res != [0, 0, 0])
mask = np.zeros(res.shape)
mask[i, j] = 1
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=3)
mask = cv2.erode(mask, kernel, iterations=3)

cv2.imwrite("res09.jpg", I * mask)
