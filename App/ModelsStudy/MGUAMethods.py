import numpy as np
import math

# Normalization -------------------------------------------------------------------------------------
def normalize(data):
    # Nested loop for multiplying the array data by their cofficients
    multipliedData = []

    # For loop with range of number of values in sublist of data set
    # (inner array) - 1 (y value which not count in normalization cycle)
    for i in range(len(data[0]) - 1):
        temp = []
        # For loop with range of number of subarrays in data set
        # array 1, array 2, ... , array n (len  of data - 1)
        for j in range(len(data)):
            secondTemp = []
            # For loop with range of number of values in sublist of data set
            # (inner array values)
            for k in range(len(data[0])):
                secondTemp.append(data[j][k] * data[j][i])
            temp.append(secondTemp)
        multipliedData.append(temp)

    # Each argument value (4) > for each of subarrays (6) > for each of values in subarray (5)
    # Firstly add to temp list then add this list to multiplied data list

    # print(np.array(multipliedData))

    # Nested loop for addition of the array set data to the singular array
    normalizedData = []

    # For loop with range of number of sublist of data set
    for i in range(len(multipliedData)):
        # List filled with 0 of sulist len
        temp = [0 for i in range(len(multipliedData[i][0]))]
        # For loop with range of number of vales from sublist
        for j in range(len(multipliedData[i])):
            # For loop with range of our empty list that will be base for values addition
            for k in range(len(temp)):
                temp[k] += multipliedData[i][j][k]
        normalizedData.append(temp)

    # print(np.array(normalizedData))

    return np.array(normalizedData)

def normalizeNew(data):
    # Nested loop for multiplying the array data by their cofficients
    multipliedData = []

    # For loop with range of number of values in sublist of data set
    # (inner array) - 1 (y value which not count in normalization cycle
    for i in range(len(data[0][1]) - 1):
        temp = []
        # For loop with range of number of subarrays in data set
        # array 1, array 2, ... , array n (len  of data - 1)
        for j in range(len(data)):
            secondTemp = []
            # For loop with range of number of values in sublist of data set
            # (inner array values)
            for k in range(len(data[0][1])):
                secondTemp.append(data[j][1][k] * data[j][1][i])
            temp.append(secondTemp)
        multipliedData.append(temp)

    # Each argument value (4) > for each of subarrays (6) > for each of values in subarray (5)
    # Firstly add to temp list then add this list to multiplied data list

    # print(np.array(multipliedData))

    # Nested loop for addition of the array set data to the singular array
    normalizedData = []

    # For loop with range of number of sublist of data set
    for i in range(len(multipliedData)):
        # List filled with 0 of sulist len
        temp = [0 for i in range(len(multipliedData[i][0]))]
        # For loop with range of number of vales from sublist
        for j in range(len(multipliedData[i])):
            # For loop with range of our empty list that will be base for values addition
            for k in range(len(temp)):
                temp[k] += multipliedData[i][j][k]
        normalizedData.append(temp)

    # print(np.array(normalizedData))

    return np.array(normalizedData)


# GAUSS METHOD ------------------------------------------------------------------------------------------------

# Gauss-Jordan
# https://www.youtube.com/watch?v=xOLJMKGNivU

# And more
# https://math.stackexchange.com/questions/3137829/solving-an-undetermined-or-overdetermined-system-of-equations-with-constraints
# https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers
# https://math.libretexts.org/Bookshelves/Linear_Algebra/Matrix_Algebra_with_Computational_Applications_(Colbry)/41%3A_21_In-Class_Assignment_-_Solve_Linear_Systems_of_Equations_using_QR_Decomposition/41.4%3A_Underdetermined_Systems

def gaussianElimination(A, y):

    if A.shape[0] != A.shape[1]:
        print("ERROR: Matrix is not square")
        raise Exception

    if y.shape[1] > 1 or y.shape[0] != A.shape[0]:
        print("ERROR: Constant values matrix is incorrectly sized")
        raise Exception

    # Vars
    n = len(A)
    m = n - 1
    i = 0
    x = np.zeros(n)

    # Create matrix
    augmentedA = np.concatenate((A, y), axis=1, dtype=float)
    # print(f"Initial augmented matrix: \n {augmentedA}")

    # Gauss elimination
    while i < n:

        # Partial pivoting
        for j in range(i + 1, n):
            if abs(augmentedA[i, i]) < abs(augmentedA[j, i]):
                augmentedA[[j, i]] = augmentedA[[i, j]]

        if augmentedA[i, i] == 0.0:
            print("ERROR: Divide by zero")
            raise Exception


        for j in range(i + 1, n):
            scaling = augmentedA[j][i] / augmentedA[i][i]
            augmentedA[j] = augmentedA[j] - (scaling * augmentedA[i])
            # print(augmentedA)
        i += 1

    # back sustitution
    x[m] = augmentedA[m][n] / augmentedA[m][m]

    for k in range(n - 2, -1, -1):
        x[k] = augmentedA[k][n]
        for j in range(k + 1, n):
            x[k] = x[k] - augmentedA[k][j] * x[j]
        x[k] = x[k] / augmentedA[k][k]

    # Result
    print("Result: \n")

    for ans in range(n):
        print(f"a{ans} = {x[ans]}")

    return x

# SEIDEL METHOD ------------------------------------------------------------------------------------------------

 # https://kishorkafle.medium.com/python-code-for-guass-seidel-method-fe948e4f46d2

def checkDiaDom(A):
    r = len(A)
    diagnl = 0
    non_diagnl = 0

    for i in range(r):
       for j in range(r):
            if i == j:
                 diagnl += abs(A[i][j])
            else:
                 non_diagnl += abs(A[i][j])

    # print('Sum of diagonal elements\n', diagnl)
    # print('Sum of non-diagonal elements\n', non_diagnl)

    if diagnl >= non_diagnl:
       verdict = True
    else:
       verdict = False
    return verdict

def GSmethod(A, Y, X):
    n = len(A)
    for j in range(0, n):
        summ_val = Y[j]
        for i in range(0, n):
            if (j != i):
                summ_val -= A[j][i] * X[i]
        X[j] = summ_val / A[j][j]
    return X

# ASSISTING METHODS ------------------------------------------------------------------------------------------------

def makePairs(x, y):
    pairs = []
    # print(x)
    # print(y)
    for i in range(len(y)):
        # for j in range(1, len(x[i])):
        #      pairs.append([y[i], [x[i][j-1], x[i][j]]])
        for j in range(len(x[i])):
            for k in range(j + 1, len(x[i])):
                pairs.append([[i, y[i]], [x[i][j], x[i][k]]])
                """Old"""
                # pairs.append([y[i], [x[i][j], x[i][k]]])
    return pairs


def makeSecondaryPairs(x):
    pairs = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pairs.append([x[i], x[j]])
    return pairs

# p = [y index,[y,[x1,x2]]]
# y = a0 + aixi + ajxj
def makeLinearFuncions(p):
    res = []
    for i in range(len(p)):
        res.append([p[i][0][0], [1, p[i][1][0], p[i][1][1], p[i][0][1]]])
        """Old"""
        # res.append([1, p[i][1][0], p[i][1][1], p[i][0]])
    return res


#  y = a0 + a1xi + a2xj + a3xi*xj + a4xi^2 + a5xj^2
def makeSquareFunctions(p):
    res = []
    for i in range(len(p)):
        res.append([p[i][0][0], [1, p[i][1][0], p[i][1][1], p[i][1][0] * p[i][1][1], p[i][1][0] ** 2, p[i][1][1] ** 2, p[i][0][1]]])
        """Old"""
        # res.append([1, p[i][1][0], p[i][1][1], p[i][1][0] * p[i][1][1], p[i][1][0] ** 2, p[i][1][1] ** 2, p[i][0]])
    return res


# xData (testing), gaussian results
def getComparisonResults(x, studyRes):
    res = []
    for i in x:
        temp = 0
        for j in range(len(i)):
            temp += studyRes[j] * i[j]

        res.append(temp)

    return res


# 1 / n * sum([Y real - Y found]**2)
def getModelUsefullnessCriteria(yCheck, yTest):
    ySum = 0
    for i in range(len(yTest)):
        ySum += (yTest[i] - yCheck[i][1]) ** 2

    return math.sqrt(ySum / len(yTest))


# Split one list into 2
def convertLists(data):
    xStudy = []
    yStudy = []
    for i in data:
        xStudy.append(i[1][:-1])
        yStudy.append([i[0],i[1][-1]])
    return xStudy, yStudy


# Split one list into 2 for Gaussian elim
def convertListsGaussian(data):
    xStudy = []
    yStudy = []
    for i in data:
        yStudy.append([i[-1]])
        xStudy.append(i[:-1])

    return xStudy, yStudy

# Make new pairs
# Req: Pairs of initial y and calculated y

# Make pairs and put each new pair to their respectteful y
# Additional limit number of new pairs

# Res: [y, [zy1,zy2]], [y, [zy2, zy3]], ...
def makeNewEquationsPairsOld(sortedPairs):
    dict = {}
    for i in sortedPairs:
        if i[0] in dict:
            t = dict[i[0]]
            t.append(i[1])
            dict[i[0]] = t
        else:
            dict[i[0]] = [i[1]]

    print(dict)
    pairs = []
    for k, v in dict.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                pairs.append([k, [v[i], v[j]]])

    print(pairs)

    return pairs

def makeNewEquationsPairs(sortedPairs):
    pairs = []
    indexes = []
    for i in sortedPairs:
        indexes.append(i[0])

    indexes = list(set(indexes))
    data = []
    for i in indexes:
        temp = []
        val = 0.0
        for j in sortedPairs:
            if j[0] == i:
                val = j[1]
                temp.append(j[2])
        data.append([i, val, temp])

    # [[0, 1.0], [2.99384509, 5.61652039]]
    for d in data:
        if len(d[2]) > 1:
            for i in range(len(d[2])):
                for j in range(i + 1, len(d[2])):
                    pairs.append([[d[0], d[1]], [d[2][i], d[2][j]]])

    return pairs

def limitSortedPairs(sortedPairs, param):
    res = []
    values = []
    for i in sortedPairs:
        values.append(i[1])

    values = list(set(values))
    for i in values:
        counter = 0
        for j in sortedPairs:
            if counter < param:
                if i == j[1]:
                    res.append(j)
                    counter += 1
            else:
                break
    return res

