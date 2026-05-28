# import math
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# import traceback, sys, os
# import numpy as np
#
# from datetime import datetime
# from sympy import symbols, Eq, solve
# from sympy.core.sympify import sympify
#
# from gmdh import Combi, split_data
#
# from App.ModelsStudy import MGUAMethods as M
#
# # https://stackoverflow.com/questions/46586520/normal-equation-implementation-in-python-numpy
# # https://dikshit18.medium.com/normal-equation-using-python-5993454fbb41
# # https://www.youtube.com/watch?v=pjYgiOE7E8o
# # https://www.c-sharpcorner.com/article/normal-equation-implementation-from-scratch-in-python/
#
# # Gauss elimination
# # https://bvanderlei.github.io/jupyter-guide-to-linear-algebra/Gaussian_Elimination.html
# # Guass-Seidel Method
# # https://kishorkafle.medium.com/python-code-for-guass-seidel-method-fe948e4f46d2#:~:text=Gauss-Seidel%20method%20is%20one,finding%20values%20of%20unknown%20variables.
# # https://www.geeksforgeeks.org/gauss-seidel-method/
# # https://www.codesansar.com/numerical-methods/python-program-gauss-seidel-iteration-method.htm
#
# # Normalization
# # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
#
# # MGUA ++++++++++++++++++++++++
# # https://pypi.org/project/gmdh/#:~:text=About%20GMDH,providing%20fast%20experimentations%20and%20studing.
#
# # Under-determinated systems +++++++++++++++++++
# # https://stackoverflow.com/questions/31098228/solving-system-using-linalg-with-constraints
#
# # https://realpython.com/python-linear-algebra/
# # https://medium.com/datadenys/solving-systems-of-linear-equations-using-matrices-and-python-5e203a9ea146
#
# # Get global params for all models ---------------------------------------------------------------------------
#
# def passParams(trainDataFeatures, trainDataLabels, testDataFeatures, testDataLabels,
#                initialFileNames, resultSavePath, initialHeaders, progresBar, consoleOutputField):
#
#      global trainFeatures, testFeatures, trainLabels, testLabels, \
#           fileNames, savePath, headers, pBar, consoleOutput
#
#      trainFeatures = trainDataFeatures
#      trainLabels = trainDataLabels
#      testFeatures = testDataFeatures
#      testLabels = testDataLabels
#      fileNames = initialFileNames
#      savePath = resultSavePath
#      headers = initialHeaders
#      pBar = progresBar
#      consoleOutput = consoleOutputField
#
#
# def getAssistantValues(resultSavePath, progresBar, consoleOutputField, selectionParam):
#
#      global savePath, pBar, consoleOutput, selectionType
#
#      savePath = resultSavePath
#      pBar = progresBar
#      consoleOutput = consoleOutputField
#      selectionType = selectionParam
#
#
# def getDataParams(trainDataFeatures, trainDataLabels, testDataFeatures, testDataLabels):
#
#      global trainFeatures, testFeatures, trainLabels, testLabels
#
#      trainFeatures = trainDataFeatures
#      trainLabels = trainDataLabels
#      testFeatures = testDataFeatures
#      testLabels = testDataLabels
#
#      # print("MODELS==============================")
#      # print("TRAIN===============================")
#      # print(trainFeatures)
#      # print(trainLabels)
#      # print("TESTING=============================")
#      # print(testFeatures)
#      # print(testLabels)
#
#
# def getDataStudyParams(trainDataFeatures, trainDataLabels):
#
#      global trainFeatures, trainLabels
#
#      trainFeatures = trainDataFeatures
#      trainLabels = trainDataLabels
#
#      # print("MODELS==============================")
#      # print("TRAIN===============================")
#      # print(trainFeatures)
#      # print(trainLabels)
#
#
# def getDataTestParams(testDataFeatures, testDataLabels):
#
#      global testFeatures, testLabels
#
#      testFeatures = testDataFeatures
#      testLabels = testDataLabels
#
#      # print("MODELS==============================")
#      # print("TESTING=============================")
#      # print(testFeatures)
#      # print(testLabels)
#
# def convertDataToAceptedFormat(labels):
#      res = []
#      for i in labels:
#           res.append(i[0])
#
#      return res
#
# # Save model results to file
# def exportModelsResult(savePath, result):
#
#     now = datetime.now()
#     date = now.strftime("%d_%m_%Y_%H_%M_%S")
#
#     res = ""
#     for i in range(len(result)):
#          res += f"a{str(i+1)} = {str(result[i])}; \n"
#          # res += f"{str(result[i])}\n"
#
#     f = open(f"D:/PythonProjects/MagRobota/ModelsResults/МГУА_{date}.txt", "a")
#     # f = open(f"{savePath}/МГУА_{date}.txt", "a")
#     f.write(res)
#     f.close()
#
# if __name__ == '__main__':
#
#      # Normalization ------------------------------------------------------------------------------------
#
#      data = [[1, 10, 15, 10 * 15, 13],
#              [1, 15, 13, 15 * 13, 19],
#              [1, 13, 19, 13 * 19, 14],
#              [1, 19, 14, 19 * 14, 18],
#              [1, 14, 18, 14 * 18, 17],
#              [1, 18, 17, 18 * 17, 11],
#              ]
#
#      # Gaussian Normalization
#      normalizedData = M.normalize(data)
#
#      # Initial and Normalizaed Data
#      print("Original Data: \n", np.array(data))
#      print("Normalized Data: \n", normalizedData)
#
#
#      # Inbuilt GMDH -------------------------------------------------------------------------------------------------
#      print("Inbuilt GMDH -------------------------------------------------------------------------------------------")
#      y = []
#      x = []
#      for i in normalizedData:
#           y.append([i[-1]])
#           x.append(i[:-1])
#
#      xTrain, xTest, yTrain, yTest = split_data(x, y)
#
#      print('X Train:\n', xTrain)
#      print('X Test:\n', xTest)
#      print('Y Train:\n', yTrain)
#      print('Y Test:\n', yTest)
#
#      # Model study
#      model = Combi()
#      model.fit(xTrain, yTrain)
#
#      # Prediction
#      yPredicted = model.predict(xTest)
#
#      print('Y Predicted: ', yPredicted)
#      print('Y Test: ', yTest)
#
#      # Best Polynominal
#      print(model.get_best_polynomial())
#
#      # LINALG SOLVING ------------------------------------------------------------------------------------------------
#      print("Least Squares Solution -------------------------------------------------------------------------------\n")
#      # Least squares
#      x = []
#      y = []
#      for i in normalizedData:
#           y.append(i[-1])
#           x.append(i[:-1])
#
#      linalgLstqRes, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
#      linalgSolveRes = np.linalg.solve(x, y)
#
#      print(linalgLstqRes)
#      print(linalgSolveRes)
#
#      print("Gaussian Solution \n")
#
#      x = []
#      y = []
#      for i in normalizedData:
#           y.append([i[-1]])
#           x.append(i[:-1])
#
#      x = np.array(x)
#      y = np.array(y)
#
#      gaussianRes = M.gaussianElimination(x, y)
#      print(gaussianRes)
#
#      # SEIDEL METHOD ------------------------------------------------------------------------------------------------
#
#      print()
#      # x = []
#      # y = []
#      # for i in normalizedData:
#      #      y.append(i[-1])
#      #      x.append(i[:-1])
#      #
#      # seidelRes = [0 for i in range(len(y))]
#      # n = len(seidelRes)
#      #
#      # print(x)
#      # print(y)
#      # print(seidelRes)
#      #
#      # if M.checkDiaDom(x) == 1:
#      #      for i in range(25):
#      #           seidelRes = M.GSmethod(x, y, seidelRes)
#      #           print(seidelRes)
#      # else:
#      #      print('Non-Diagonal matrix')
#      print()
#
#      # EQUASION METHOD ------------------------------------------------------------------------------------------------
#
#      # print("Equation solution ---------------------------------------------------------------------------------------")
#      x = []
#      y = []
#      for i in normalizedData:
#           y.append(i[-1])
#           x.append(i[:-1])
#
#      def generateEquasion(x, y):
#
#           globalStringEquasion = []
#           globalListEquasion = []
#           for i in range(len(y)):
#                equasionList = ["%s" % y[i]]
#                equasionString = ""
#                for j in range(len(x[i])):
#                     equasionString += "a%d * %d + " % (j+1, x[i][j])
#                     equasionList.append("a%d * %d" % (j+1, x[i][j]))
#
#                if equasionString[-1] == "+" or equasionString[-2] == "+":
#                     equasionString = equasionString[:-3]
#
#                globalStringEquasion.append(Eq(sympify(equasionString), y[i]))
#                globalListEquasion.append(equasionList)
#
#
#           return globalStringEquasion #, globalListEquasion
#
#      print("Equations")
#      print(generateEquasion(x, y))
#
#      keys = {}
#      for i in range(len(x[0])+1):
#           keys["a%d" % i] = symbols("a%d" % i)
#
#      keysList = []
#      for v in keys.values():
#           keysList.append(v)
#
#      print(solve((generateEquasion(x, y)), keysList))
#
#      z = solve((generateEquasion(x, y)), keysList)
#
#      equationRes = []
#      for k, v in z.items():
#           equationRes.append(float(v))
#
#      print(equationRes)
#
#      # QR DECOMPOSITION -----------------------------------------------------------------------------------------------
#      A = np.array(normalizedData)
#      Q, R = np.linalg.qr(A)
#
#      # RESULTS CHECK --------------------------------------------------------------------------------------------------
#
#      # Check data set(array)
#      # xInitial = []
#      # yInitial = []
#      # for i in data:
#      #      yInitial.append(i[-1])
#      #      xInitial.append(i[:-1])
#      #
#      # # print("X and Y ")
#      # # print(xInitial)
#      # # print(yInitial)
#      #
#      # res = []
#      # for i in xInitial:
#      #      temp = 0
#      #      for j in range(len(i)):
#      #           # linalgLstqRes
#      #           # linalgSolveRes
#      #           # gaussianRes
#      #           # seidelRes
#      #           # equationsRes
#      #           temp += gaussianRes[j] * i[j]
#      #      res.append(temp)
#
#      # print(res)
#
#      # def studyModel():
#      def studyModel(xTrain, yTrain, xTest, yTest):
#
#           """ 0 - Split x and y into Study and testing sets"""
#
#           firstCycle = True
#           criteria = 0
#           cycle = 1
#
#           """ Swap if working with the interface """
#           # xStudy, yStudy = trainFeatures, trainLabels
#           # xTest, yTest = testFeatures, testLabels
#
#           selectionType = 1
#           # xStudy, yStudy = x, y
#           # xTest, yTest = [], []
#
#           xStudy, yStudy = xTrain, yTrain
#           xTest, yTest = xTest, yTest
#
#           yStudy, yTest = convertDataToAceptedFormat(yStudy), convertDataToAceptedFormat(yTest)
#
#           xlen = len(xStudy[0]) * len(xStudy)
#
#           """ 1 - Make pairs """
#           print("1 - Пари \n")
#
#           pairs = M.makePairs(xStudy, yStudy)
#           print(pairs)
#
#           testPairs = M.makePairs(xTest, yTest)
#           print(testPairs)
#
#           """ 2 - Make equations """
#           print("2 - Умовні рівняння \n")
#
#           # linear = M.makeLinearFuncions(pairs)
#           # square = M.makeSquareFunctions(pairs)
#           linear = M.makeSquareFunctions(pairs)
#
#           print(linear)
#           # print(square)
#
#           # linearTest = M.makeLinearFuncions(testPairs)
#           # squareTest = M.makeSquareFunctions(testPairs)
#           linearTest = M.makeSquareFunctions(testPairs)
#
#           print(linearTest)
#           # print(squareTest)
#
#           """ 3 - Find Coeff`s (a0, a1, ... , an) """
#           print("3.1 - Нормалізація \n")
#
#           # normalizedData = M.normalize(linear)
#           # normalizedDataSquare = M.normalize(square)
#
#           normalizedData = M.normalizeNew(linear)
#
#           print(normalizedData)
#           # print(normalizedDataSquare)
#
#           # normalizedTestData = M.normalize(linearTest)
#           # normalizedTestDataSquare = M.normalize(squareTest)
#
#           normalizedTestData = M.normalizeNew(linearTest)
#
#           xStudyNormalized, yStudyNormalized = M.convertListsGaussian(normalizedData)
#           xStudyNormalized = np.array(xStudyNormalized)
#           yStudyNormalized = np.array(yStudyNormalized)
#
#           xTestNormalized, yTestNormalized = M.convertListsGaussian(normalizedTestData)
#           xTestNormalized = np.array(xTestNormalized)
#           yTestNormalized = np.array(yTestNormalized)
#
#           print("3.2 - Коефіцієнти \n")
#           # Add additional methods (Siedel, linear, equations...)
#           gaussianRes = M.gaussianElimination(xStudyNormalized, yStudyNormalized)
#           res = gaussianRes
#
#           """ Find Y values on testing set of data """
#           print("4.1 - Перевірка результатів \n")
#
#           gaussianCheckRes = M.getComparisonResults(xStudyNormalized, gaussianRes)
#           print(gaussianCheckRes)
#
#           gaussianCheckResTest = M.getComparisonResults(xTestNormalized, gaussianRes)
#           print(gaussianCheckResTest)
#
#           # Pairs Check
#           xLinear, yLinear = M.convertLists(linear)
#           initialLinear = yLinear
#
#           print(linear)
#           print(xLinear)
#           print(yLinear)
#
#           xTestLinear, yTestLinear = M.convertLists(linearTest)
#
#           print("4.2 - Перетворення масиву лін. рівнянь до звичайного вигляду \n")
#
#           """ Замінити у-ки, що використовувалися для навчання та у-ки для тестування """
#           """ Res = M.getComparisonResults(xTestNormalized, coefficients) """
#
#           gaussianCheckRes = M.getComparisonResults(xLinear, gaussianRes)
#
#           print(gaussianCheckRes)
#
#           gaussianCheckResTest = M.getComparisonResults(xTestLinear, gaussianRes)
#
#           print(gaussianCheckResTest)
#
#           print("4.3 - Переіврка результату за Гауссом (Моделювання у порівняння з оригіналами) \n")
#
#           for i in range(len(gaussianCheckRes)):
#                print("Index %d, value %d - prediction %f" % (yLinear[i][0], yLinear[i][1], gaussianCheckRes[i]))
#
#           print("\n")
#
#           for i in range(len(gaussianCheckResTest)):
#                print("Index %d, value %d - prediction %f" % (yLinear[i][0], yLinear[i][1], gaussianCheckResTest[i]))
#
#           # print(yLinear)
#           # print(gaussianCheckRes)
#
#           """ Переглянути критерій """
#
#           criteria = M.getModelUsefullnessCriteria(yLinear, gaussianCheckRes)
#
#           criteriaTest = M.getModelUsefullnessCriteria(yTestLinear, gaussianCheckResTest)
#
#           print("5 - Пошук критерію \n")
#           print(criteria)
#           print(criteriaTest)
#
#           print("ЦИКЛ: %d \n" % cycle)
#           cycle += 1
#
#           # Next Steps ------------------------------------------------------------------------------------------------
#           """ New cycle """
#
#           newCriteria = criteria
#           newRes = res
#
#           # Check if last criteria value < current one; if True: another step, else: break
#
#           """ Loop Cycles """
#
#           while newCriteria <= criteria:
#
#                print("6 - Циклічний метод \n")
#
#                """ Set newCriteria as current one"""
#                criteria = newCriteria
#                res = newRes
#
#                # Get pairs y and y initial with smallest differenece
#                # Req: Initial y values, Calculated y values
#                checkPairs = []
#                checkTestPairs = []
#
#                initialLinear = yLinear
#
#                print(len(initialLinear))
#                print(initialLinear)
#                print(gaussianCheckRes)
#
#                """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
#
#                """
#                   Sort models my criteria with their corresponding indexes
#                   Create new pairs without messing their indexes and corresponding initial y values
#                """
#
#                temp1 = []
#                for i in range(len(gaussianCheckResTest)):
#                     temp1.append([yTestLinear[i][0], yTestLinear[i][1], gaussianCheckResTest[i]])
#                     # temp1.append([yLinear[i][0], yLinear[i][1], gaussianCheckRes[i]])
#
#                print(temp1)
#
#                sortedPairs = sorted(temp1, key=lambda sub: abs(sub[2] - sub[1]))
#                print(sortedPairs)
#
#                for i in range(len(sortedPairs)):
#                     print("Index %d, value %d - prediction %f" % (
#                     sortedPairs[i][0], sortedPairs[i][1], sortedPairs[i][2]))
#
#                print(len(sortedPairs))
#
#                """
#                    Sort models my criteria with their corresponding indexes
#                    Create new pairs without messing their indexes and corresponding initial y values
#
#                    Get n best models (n decided by input in interface) from test data
#                    Create new pairs for n models with indexes of best models from prev test
#
#                """
#
#                """ Use gaussianCheckResTest """
#
#                # if len(initialLinear) == len(gaussianCheckRes):
#                #      for i in range(len(initialLinear)):
#                #           checkPairs.append((initialLinear[i], gaussianCheckRes[i]))
#                #
#                # # if len(initialLinear) == len(gaussianCheckResTest):
#                # #      for i in range(len(initialLinear)):
#                # #           checkPairs.append((initialLinear[i], gaussianCheckResTest[i]))
#                #
#                # sortedPairs = sorted(checkPairs, key=lambda sub: abs(sub[1] - sub[0]))
#                # print(len(sortedPairs))
#                # print(sortedPairs)
#
#                """ Number of models for next layer """
#
#                # if selectionType == 0:
#                #      sortedPairs = sortedPairs[:xlen]
#                # elif selectionType == 1:
#                #      if len(sortedPairs) > 100:
#                #           sortedPairs = sortedPairs[:len(sortedPairs)//4]
#                #      elif len(sortedPairs) > 10000:
#                #           sortedPairs = sortedPairs[:len(sortedPairs)//40]
#                #      elif len(sortedPairs) > 100000:
#                #           sortedPairs = sortedPairs[:len(sortedPairs)//400]
#                # else:
#                #      sortedPairs = sortedPairs[:selectionType]
#
#                """ All actions for limiting the number of models inside this method"""
#                sortedPairs = M.limitSortedPairs(sortedPairs, 15)
#
#                # sortedPairs = sorted(sortedPairs, key=lambda sub: abs(sub[2] - sub[1]))
#                #
#                # for i in range(len(sortedPairs)):
#                #      print("Index %d, value %d - prediction %f" % (
#                #           sortedPairs[i][0], sortedPairs[i][1], sortedPairs[i][2]))
#
#                # if len(sortedPairs) > 100:
#                #      sortedPairs = sortedPairs[:100]
#                #
#                # sortedPairs = sortedPairs[:30]
#
#                print(len(sortedPairs))
#                print(sortedPairs)
#
#                print(sorted(sortedPairs, key=lambda sub: sub[0]))
#
#                yPairs = M.makeNewEquationsPairs(sortedPairs)
#                print(len(yPairs))
#
#                print("6.1 - Нові рівняння (Аргументи) із нових пар \n")
#                # linear = M.makeLinearFuncions(yPairs)
#                linear = M.makeSquareFunctions(yPairs)
#                print(linear)
#
#                normalizedData = M.normalizeNew(linear)
#                # print(normalizedData)
#
#                xStudy, yStudy = M.convertListsGaussian(normalizedData)
#
#                xStudy = np.array(xStudy)
#                yStudy = np.array(yStudy)
#
#                print("6.2 - Нові коефіцієнти \n")
#                try:
#                     gaussianRes = M.gaussianElimination(xStudy, yStudy)
#                     newRes = gaussianRes
#                except Exception:
#                     return res
#
#                print(newRes)
#
#                # Pairs Check
#                xLinear, yLinear = M.convertLists(linear)
#                print(xLinear)
#
#
#                gaussianCheckRes = M.getComparisonResults(xLinear, gaussianRes)
#                gaussianCheckResTest = M.getComparisonResults(xTestLinear, gaussianRes)
#
#                print("6.3 - Новий результат перевірки \n")
#
#
#
#                for i in range(len(gaussianCheckRes)):
#                     print("Index %d, value %d - prediction %f" % (yLinear[i][0], yLinear[i][1], gaussianCheckRes[i]))
#
#                print("\n")
#
#                for i in range(len(gaussianCheckResTest)):
#                     print("Index %d, value %d - prediction %f" % (yTestLinear[i][0], yTestLinear[i][1], gaussianCheckResTest[i]))
#
#
#
#                # checkRes = []
#                # for i in range(len(gaussianCheckRes)):
#                #      print("%d - %.15f" % (yLinear[i], gaussianCheckRes[i]))
#                #      checkRes.append([yLinear[i], gaussianCheckRes[i]])
#                # # print(checkRes)
#                #
#                # checkTestRes = []
#                #
#                # print(xTestLinear)
#                # print(yTestLinear)
#                # for i in range(len(gaussianCheckResTest)):
#                #      print("%d - %f" % (yTestLinear[i], gaussianCheckResTest[i]))
#                #      checkTestRes.append([yTestLinear[i], gaussianCheckResTest[i]])
#                # print(checkTestRes)
#
#                newCriteria = M.getModelUsefullnessCriteria(yLinear, gaussianCheckRes)
#                newCriteriaTest = M.getModelUsefullnessCriteria(yTestLinear, gaussianCheckResTest)
#
#                print("6.4 - Новий критерій")
#                print(newCriteria)
#                print(newCriteriaTest)
#                print(criteria > newCriteria)
#
#                print("ЦИКЛ: %d \n" % cycle)
#                cycle += 1
#
#           """
#
#           What we return ?
#
#           """
#           # exportModelsResult(savePath, res)
#           exportModelsResult("savePath", res)
#           return res
#
# # # # Global array of parameters x
# # xTrain = data = [[11, 13, 15, 16],
# #                 [3, 4, 5, 6],
# #                 [2, 3, 1, 1],
# #                 [14, 12, 11, 17],
# #                 [3, 1, 2, 3],
# #                 [12, 11, 14, 12],
# #                 ]
# # # Global array of values y
# # yTrain = [2, 1, 1, 2, 1, 2]
#
# xTrain = [[2.99384509, 5.61652039, 7.89287302, 2.74016928, 5.36355908, 5.65949121],
#           [2.38442435, 1.71141728, 2.03163108, 7.66540101, 2.35957891, 5.35536608],
#           [4.92512384, 1.39248137, 7.56299626, 3.4261539,  5.37462751, 0.99756968],
#           [0.35260258, 0.31705686, 0.12702599, 0.65863322, 0.11808968, 0.41696986],
#           [0.26137345, 0.38886006, 0.29720913, 0.79073445, 1.20268626, 0.60634808]]
#
# yTrain = [[1.], [1.], [1.], [-1.], [-1.]]
#
# xTest = [[4.67786784, 5.16590572, 3.49928653, 12.58824546, 5.06273157, 3.57947524],
#          [1.94779559, 3.55085965, 4.3488971, 6.08857071, 3.37817007, 5.84836253],
#          [4.17547099, 1.02650897, 2.1558139, 3.50214254, 1.74762023, 2.59013977],
#          [0.23389277, 0.1410945, 0.8470022, 0.3038105, 0.35851895, 0.46083974],
#          [1.10787073, 0.31255823, 0.25736665, 1.04195842, 0.77453465, 0.44858372]]
#
# yTest = [[1.], [1.], [1.], [-1.], [-1.]]
#
# crit = studyModel(xTrain, yTrain, xTest, yTest)
# print(crit)
#
#
#
#      # if firstCycle == False:
#      #
#      #      """ Make NEW Y pairs """
#      #      # Y Pairs
#      #
#      #      #  Во втором ряду
#      #      # алгоритма полученные на обучающей выборке значения y , соответствующие отобранным моделям,
#      #      # рассматриваются в качестве аргументов нового ряда, Здесь функции f остаются прежними и соответствуют соотношениям
#      #      # но аргументами этих функций выступают переменные y .Коэффициенты новых моделей находятся по МНК на точках той
#      #      # же обучающей последовательности. Новые модели проверяются на точках проверочной последовательности, и среди них
#      #      # выбирается S наилучших, которые используются в качестве аргументов следующего третьего ряда и т.д.
#      #
#      #      # I should use y-pairs as an arguments to already created linear functions >
#      #      # Used y argument - new pairs
#      #      # Chose n best and retry
#      #
#      #      # I Create n pairs from y results of previous layers
#      #      # y = 10
#      #      # > 45 pairs
#      #      # Lower number of pairs, then create new linear functions
#      #      # We need to lower number of pairs and choose best of them
#      #      # 1) Sort created pairs: select pairs that will be the closest to original y
#      #      # 2) When creating pairs limit number
#      #
#      #      # Get pairs y and y initial with smallest differenece
#      #      # Req: Initial y values, Calculated y values
#      #      checkPairs = []
#      #      """ Відсортувати та перевірити пари так, щоб оригінальне значення знаходилося в парі з прогнозованим """
#      #      if len(initialLinear) == len(gaussianCheckRes):
#      #           for i in range(len(initialLinear)):
#      #                checkPairs.append((initialLinear[i], gaussianCheckRes[i]))
#      #
#      #      print("Перша перевірка пар")
#      #      print(checkPairs)
#      #
#      #      """ Сортування по мінімальній різниці між оригінальним і прогнозованим значенням """
#      #
#      #      sortedPairs = sorted(checkPairs, key=lambda sub: abs(sub[1] - sub[0]))
#      #      # print(sortedPairs)
#      #      """ Зменшити кількість пар, залишивши ли ті, що мають найменше відхилення """
#      #      if len(sortedPairs) > 100:
#      #           sortedPairs = sortedPairs[:100]
#      #
#      #      sortedPairs = sortedPairs[:len(sortedPairs) // 2]
#      #
#      #      # Additional control minimum number of 1 and 2
#      #
#      #      # Make new pairs
#      #      # Req: Pairs of initial y and calculated y
#      #
#      #      # Make pairs and put each new pair to their respectteful y
#      #      # Additional limit number of new pairs
#      #
#      #      # Res: [y, [zy1,zy2]], [y, [zy2, zy3]], ...
#      #      print(sortedPairs)
#      #
#      #      """ !!!!!!!!!!!!! """
#      #
#      #      print("6 - Нові пари (Із знайдених у-ків)")
#      #      yPairs = M.makeNewEquationsPairs(sortedPairs)
#      #      # yPairs = M.makeNewEquationsPairs(checkPairs)
#      #
#      #      # print(yPairs)
#      #
#      #      """  Repeat Cycle """
#      #
#      #      print("7 - Повтор циклу навчання \n")
#      #
#      #      linear = M.makeLinearFuncions(yPairs)
#      #      print(linear)
#      #
#      #      normalizedData = M.normalize(linear)
#      #      # print(normalizedData)
#      #
#      #      xStudy, yStudy = M.convertListsGaussian(normalizedData)
#      #
#      #      xStudy = np.array(xStudy)
#      #      yStudy = np.array(yStudy)
#      #
#      #      gaussianRes = M.gaussianElimination(xStudy, yStudy)
#      #
#      #      # Pairs Check
#      #      xLinear, yLinear = M.convertLists(linear)
#      #
#      #      gaussianCheckRes = M.getComparisonResults(xLinear, gaussianRes)
#      #
#      #      print("7.1 - Результат за Гауссом \n")
#      #
#      #
#      #      for i in range(len(gaussianCheckRes)):
#      #           print("%d - %f" % (yLinear[i], gaussianCheckRes[i]))
#      #
#      #      # print(yLinear)
#      #      # print(gaussianCheckRes)
#      #
#      #      newCriteria = M.getModelUsefullnessCriteria(yLinear, gaussianCheckRes)
#      #
#      #      print("7.2 - Новий критерій")
#      #      print(newCriteria)
#      #      firstCycle = False
#      #
#      #      print("ЦИКЛ: %d \n" % cycle)
#      #      cycle += 1
#
#
#      # 7 - Use solved (found) y values for new pairs?
#
#      # 8 - Repeat steps 2-6, if ocinka getting smaller we proceed, else: stop
#
#      # Stduy our algorithms via selected methods
#      # Check results via putting parameters from previous results into initial formula
#      # x1 * 1 + x2 * 2 + x3 * 3 + ... + xn * n
#      # =
#      # 0.1 * 1 + 0.2 * 2 + 0.3 * 3 + ... + 0.n * n
#      # Compare initial y with result
#
#      # Kriteriy korisnosti modeli
#      # Vrode kak: Sy = sqrt(sum((yi - yi(initial)) ^ 2)/n)
#
#      # Best model is the model with smallest Sy
#
#
#
#      # https://stackoverflow.com/questions/54715366/solving-system-of-equations-in-python
#      # https://problemsolvingwithpython.com/05-NumPy-and-Arrays/05.08-Systems-of-Linear-Equations/
#      # https://www.geeksforgeeks.org/python-solve-the-linear-equation-of-multiple-variable/