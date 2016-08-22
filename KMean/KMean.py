from numpy import *


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centers = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centers[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centers


def kMeans(dataSet, k, distMethod=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssess = mat(zeros((m, 2)))
    centers = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each sample
            # get closest center
            minDist = inf
            minIndex = -1
            for j in range(k):  # for each class
                dist = distMethod(centers[j, :], dataSet[i, :])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssess[i, 0] != minIndex:
                clusterChanged = True
            clusterAssess[i, :] = minIndex, minDist**2
        # update center
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssess[:, 0].A==cent)[0]]
            centers[cent,:] = mean(ptsInClust, axis=0)
    return centers, clusterAssess

datMat = mat(loadDataSet('testSet.txt'))
center, clust = kMeans(datMat, 4)
print(clust)