import numpy as np
import cv2

def drawContour(img, points, markersize, type):
    imgDraw = img.copy()
    if (type == 'l' or type == 'p&l'):
        for p in range(len(points)):
            point1 = points[p]
            point2 = points[(p - 1) % len(points)]
            imgDraw = cv2.line(imgDraw, (point1[1], point1[0]), (point2[1], point2[0]), color=(0, 0, 255),
                               thickness=markersize)
    if (type == 'p' or type == 'p&l'):
        for p in range(len(points)):
            imgDraw = cv2.circle(imgDraw, (points[p][1], points[p][0]), markersize, color=(0, 0, 155), thickness=-1)
    return imgDraw


def calAvgDist(points):
    dists = []
    for p in range(len(points)):
        point1 = points[p]
        point2 = points[(p - 1) % len(points)]
        dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** (1 / 2)
        dists.append(dist)
    return np.mean(dists)


def externalE(pointCur):
    global gamma
    global gradient
    E = -gamma * (gradient[pointCur[0], pointCur[1]] ** 2)
    return E


def internalE(pointPrv, pointCur, dbar):
    global alpha
    E = alpha * abs((pointCur[0] - pointPrv[0]) ** 2 + (pointCur[1] - pointPrv[1]) ** 2 - dbar ** 2) ** 2
    return E


def centralE(pointCur):
    global beta
    global center
    E = beta * ((center[0] - pointCur[0]) ** 2 + (center[1] - pointCur[1]) ** 2)
    return E


def energy(pointPrv, pointCur, dbar):
    return internalE(pointPrv, pointCur, dbar) + externalE(pointPrv) + centralE(pointPrv)


def iterate(points):
    neighbors = np.array([[0, 0], [0, -1], [0, 1], [1, 0], [1, -1], [1, 1], [-1, 0], [-1, -1], [-1, 1]])
    energies = np.ones((len(points), len(neighbors), len(neighbors))) * float('inf')
    positions = np.zeros((len(points), len(neighbors)), dtype=int)
    dbar = calAvgDist(points)
    for p in range(len(points)):
        pointCur = points[p]
        pointPrv = points[(p - 1) % len(points)]
        for i in range(len(neighbors)):
            pointCurNew = pointCur + neighbors[i]
            minE = float('inf')
            for j in range(len(neighbors)):
                pointPrvNew = pointPrv + neighbors[j]
                try:
                    E = energy(pointPrvNew, pointCurNew, dbar)
                    energies[p, i, j] = E
                    if (E < minE):
                        minE = E
                        positions[p, i] = j
                except:
                    continue

    minTotalE = float('inf')
    bestStartPose = 0
    for i in range(len(neighbors)):
        totalE = 0
        poseCur = i
        for p in range(len(points) - 1, 0, -1):
            posePrv = positions[p][poseCur]
            totalE = totalE + energies[p, poseCur, posePrv]
            poseCur = posePrv
        totalE = totalE + energies[0, poseCur, i]
        if (totalE < minTotalE):
            minTotalE = totalE
            bestStartPose = i

    poseCur = bestStartPose
    for p in range(len(points) - 1, -1, -1):
        points[p][0] = points[p][0] + neighbors[poseCur, 0]
        points[p][1] = points[p][1] + neighbors[poseCur, 1]
        poseCur = int(positions[p, poseCur])
    return points


def click(event, x, y, p1, p2):
    global points
    global Iclick
    global status
    global center
    if (event == cv2.EVENT_LBUTTONDOWN):
        if (status == "first click"):
            center = [y, x]
            Iclick = cv2.circle(Iclick, (x, y), radius=6, color=(0, 0, 0), thickness=-1)
            Iclick = cv2.putText(Iclick, "center", (x-25,y-15),cv2.FONT_ITALIC, 0.5,(0,0,0),1)
            cv2.imshow("tasbih", Iclick)
            status = "not first click"
        else:
            points.append([y, x])
            Iclick = cv2.circle(Iclick, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("tasbih", Iclick)


gamma = 100
alpha = 0.04
beta = 1


Iclick = cv2.imread("tasbih.jpg")
points = []
status = "first click"
center = [0, 0]
cv2.imshow("tasbih", Iclick)
cv2.setMouseCallback("tasbih", click)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("you selected", str(len(points) + 1), "points\nloading... ", end='')
I = cv2.imread("tasbih.jpg")
height, width, channels = I.shape
gradient = cv2.Canny(I, 100, 150)
kernel = np.ones((10, 10), np.uint8)
gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
gradient = cv2.Canny(gradient, 100, 150)

imagesList = []
for k in range(400):
    points = iterate(points)
    # center = np.mean(points, axis=0).astype(int)
    imagesList.append(cv2.circle(drawContour(I, points, 4, 'p&l'), (center[1], center[0]), 4, (0, 255, 0), -1))

out = cv2.VideoWriter("contour.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))
for i in range(len(imagesList)):
    out.write(imagesList[i])
out.release()
cv2.imwrite("res10.jpg", drawContour(I, points, 4, 'l'))
print("\rfinished")
