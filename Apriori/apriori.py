from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createUnit(dataSet):  # create cell with one element
    universe = []
    for cell in dataSet:
        for item in cell:
            if not [item] in universe:
                universe.append([item])
    return map(frozenset, universe)


def filterCandidates(dataSet, candidates, limit):
    cellCount = {}
    for cell in dataSet:
        for candidate in candidates:
            if candidate.issubset(cell):
                if not candidate in cellCount:
                    cellCount[candidate] = 1
                else:
                    cellCount[candidate] += 1
    cellNum = len(dataSet)
    selected = []
    supports = {}
    for cell in cellCount:
        support = float(cellCount[cell]) / cellNum
        if support >= limit:
            selected.insert(0, cell)
        supports[cell] = support
    return selected, supports


def createKCell(origins, k):
    cells = []
    originCount = len(origins)
    for i in range(originCount):
        for j in range(i + 1, originCount):
            list1 = list(origins[i])[:k - 2]
            list2 = list(origins[j])[:k - 2]
            list1.sort()
            list2.sort()
            if list1 == list2:  # if first k-2 elements are equal
                cells.append(origins[i] | origins[j])  # set union
    return cells


def apriori(dataMat, limit=0.5):
    units = createUnit(dataMat)
    dataSet = map(set, dataMat)
    origin, supports = filterCandidates(dataSet, units, limit)
    candidates = [origin]
    k = 2
    while (len(candidates[k - 2]) > 0):
        cellK = createKCell(candidates[k - 2], k)
        cellK, supportK = filterCandidates(dataSet, cellK, limit)
        supports.update(supportK)
        candidates.append(cellK)
        k += 1
    return candidates, supports


if __name__ == '__main__':
    dataSet = loadDataSet()
    # units = createUnit(dataSet)
    # print(units)
    # selected, supports = filterCandidates(dataSet, units, 0.5)
    # print(selected, supports)
    # cells = createKCell(selected, 2)
    # print(cells)
    selected, supports = apriori(dataSet, 0.5)
    print(selected)
