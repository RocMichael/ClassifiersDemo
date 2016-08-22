import KMean
from numpy import *


class KMeanCluster:
    def __init__(self):
        self.centers = []
        self.dataMat = []

    def loadDataSet(self, filename):
        dataMat = []
        fr = open(filename)
        for line in fr.readlines():
            curLine = line.strip().split()
            fltLine = map(float, curLine)
            dataMat.append(fltLine)
        self.dataMat = mat(dataMat)
        return self.dataMat

    def train(self, k=4):
        centers, assess = KMean.binKMeans(self.dataMat, k)
        self.centers = centers
        print(centers)

    def test(self):
        dataMat = self.loadDataSet('testSet.txt')
        centers, clust = KMean.kMeans(dataMat, 4)
        print(centers)


if __name__ == '__main__':
    kmean = KMeanCluster()
    kmean.test()
