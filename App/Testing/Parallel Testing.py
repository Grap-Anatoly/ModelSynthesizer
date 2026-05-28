# from multiprocessing import Process
# # import time
# # import Functions as f
# #
# # import joblib
# # from joblib import Parallel, delayed
# #
# #
# def func1():
#   print('func1: starting')
#   for i in range(100000):
#     print("F1 = %d" % i)
#     pass
#   print('func1: finishing')
#
#
# def func2():
#   print('func2: starting')
#   for i in range(100000):
#     print("F2= %d" % i)
#     pass
#   print('func2: finishing')
#
# if __name__ == '__main__':
#     process1 = Process(target=func1)
#     process2 = Process(target=func2)
#
#     process1.start()
#     process2.start()
#
#     process1.join()
#     process2.join()

import os
import shutil
saveFilePath = 'D://PythonProjects//MagRobota//data//TestingDump//Test//Models//ParalelTest'
models = os.listdir('D:\PythonProjects\MagRobota\data\TestingDump\Test\Models\ParalelTest')
print(models)

files = {}


for model in models:
    before, sep, after = model.rpartition(", ")
    files[before] = after
    for key, value in files.items():
        if after > value and key != before:
            files[before] = after

bestModels = []
for key, value in files.items():
    bestModels.append(key + ", " + value)

for filename in os.listdir(saveFilePath):
    if filename not in bestModels:
        shutil.rmtree(saveFilePath + "/" + filename, ignore_errors=False, onerror=None)

print(bestModels)
