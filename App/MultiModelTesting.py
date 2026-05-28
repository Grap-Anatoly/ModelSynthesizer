from abc import ABC, abstractmethod
import os
import tensorflow as tf
import torch
import numpy as np
import time
import psutil
import pynvml

if __name__ == '__main__':

    x_test = np.load("D:/PythonProjects/MagRobota/processed_dataset/X_test.npy")
    y_test = np.load("D:/PythonProjects/MagRobota/processed_dataset/y_test.npy")

    models_path = "D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/1_Paralelno/MNIST_Conv/"

    process = psutil.Process(os.getpid())

    def get_cpu_ram():
        cpu = psutil.cpu_percent(interval=0.1)
        ram = process.memory_info().rss / (1024 ** 2)  # MB
        return cpu, ram

    def useModel(x_test, y_test, modelPath):

        testModel = tf.keras.models.load_model(modelPath)
        res = []

        testModel.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        loss, acc = testModel.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(testModel.predict(x_test[:30]), axis=1)

        print(f"Model: {testModel.name}")
        print(f"Model loss: {loss}")
        print(f"Model accuracy: {acc}")
        print(f"Predicted results \n : {y_pred}")

        hit = 0
        for i in range(30):
            if y_pred[i] == y_test[i]:
                hit += 1

        predictedPerc = (hit / 30) * 100
        print(hit)
        print(f"Predicted to True acc: {predictedPerc}%")

        return acc, loss, predictedPerc

    res = []

    """Paralel testing - Models one by one"""

    print("Paralel\n")

    start = time.time()
    cpu_before, ram_before = get_cpu_ram()

    accuracy, usageloss, percentage = 0, 0, 0
    modelsNumber = 0
    bestAcc = 0

    for folder_name in os.listdir(models_path):
        folder_path = os.path.join(models_path, folder_name)
        # Only folders
        if not os.path.isdir(folder_path):
            continue

        # Check if it's a TensorFlow SavedModel
        model_file = os.path.join(folder_path, "saved_model.pb")

        if not os.path.isfile(model_file):
            print(f"Skipped (no model): {folder_name}")
            continue

        try:
            modelsNumber += 1
            acc, loss, perc = useModel(x_test, y_test, folder_path)
            accuracy += acc
            usageloss += loss
            percentage += perc

            if acc > bestAcc:
                bestAcc = acc

        except Exception as e:
            print(f"Error loading {folder_name}: {e}")

    end = time.time()
    cpu_after, ram_after = get_cpu_ram()

    print(f"\n Time per cycle: {end - start:.3f} sec")
    print(f"CPU and RAM usage: {cpu_after, ram_after}")

    print(f"Avg. Acc: {accuracy / modelsNumber}")
    print(f"Avg. Loss: {usageloss / modelsNumber}")
    print(f"Avg. Perc: {percentage / modelsNumber}")
    print(f"Best Model Acc: {bestAcc}")

    """Multilayer - Cycles of model testing
       15 models - testing, 10 of the best in the next round, 5 is deleted >
       10 models - testing, 5 of the best in the next round, 5 is deleted >
       5 models - testing, 2 of the best in the next round, 3 is deleted >
       2 models - testing, 1 of the best 1 is deleted >
       
       result = best acc, time spent, avg acc, avg loss, cpu, ram
    """
    def get_model_paths(base_dir):
        paths = []

        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)

            if os.path.isfile(os.path.join(folder_path, "saved_model.pb")):
                paths.append(folder_path)

        return paths



    def run_tournament(model_paths, X_test, y_test, stages):
        current_models = model_paths.copy()
        accuracy, usageloss, percentage = 0, 0, 0
        bestAcc = 0

        for stage_idx, keep_n in enumerate(stages):
            print(f"\n=== Stage {stage_idx + 1} ===")
            print(f"Models in: {len(current_models)}")

            results = []

            for path in current_models:
                print(f"Testing: {path}")

                model = tf.keras.models.load_model(path, compile=False)
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )

                loss, acc = model.evaluate(X_test, y_test, verbose=0)

                results.append({
                    "path": path,
                    "accuracy": acc
                })

                print(f"Accuracy: {acc}")
                print(f"Loss: {loss}")

                y_pred = np.argmax(model.predict(x_test[:30]), axis=1)
                hit = 0
                for i in range(30):
                    if y_pred[i] == y_test[i]:
                        hit += 1

                predictedPerc = (hit / 30) * 100

                accuracy += acc
                usageloss += loss
                percentage += predictedPerc

                del model

            # Sort by accuracy (descending)
            results.sort(key=lambda x: x["accuracy"], reverse=True)

            # Keep top N
            current_models = [r["path"] for r in results[:keep_n]]

            # Log results
            print("\nTop models:")
            for r in results[:keep_n]:
                print(f"{r['path']}, Accuracy: {r['accuracy']:.4f}")
                if r['accuracy'] > bestAcc:
                    bestAcc = r['accuracy']

            print("\nEliminated:")
            for r in results[keep_n:]:
                print(f"{r['path']}, Accuracy: {r['accuracy']:.4f}")

        return current_models, bestAcc, accuracy, usageloss, percentage


    STAGES = [
        10,  # from 15 → keep 10
        5,  # from 10 → keep 5
        2,  # from 5 → keep 2
        1  # final winner
    ]

    model_paths = get_model_paths("D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/1_Paralelno/MNIST_Conv/")

    print("Tournament\n")

    start = time.time()
    cpu_before, ram_before = get_cpu_ram()

    bestAcc = 0
    winner, bestAcc, accuracy, usageloss, percentage = run_tournament(
        model_paths,
        x_test,
        y_test,
        STAGES,
    )

    end = time.time()
    cpu_after, ram_after = get_cpu_ram()

    print("\n Best model:", winner)

    print(f"\n Time per cycle: {end - start:.3f} sec")
    print(f"CPU and RAM usage: {cpu_after, ram_after}")

    print(f"Avg. Acc: {accuracy / 32}")
    print(f"Avg. Loss: {usageloss / 32}")
    print(f"Avg. Perc: {percentage / 32}")
    print(f"Best Model Acc: {bestAcc}")

    """Tree - layered structure 
       object > 1 layer
       model type > 2 layer
       layers numer > 3 layer
       (layers type > 4 layer)
       
       result = best acc, time spent, avg acc, avg loss, cpu, ram
    """

    x_test = np.load("D:/PythonProjects/MagRobota/processed_dataset/X_test.npy")
    y_test = np.load("D:/PythonProjects/MagRobota/processed_dataset/y_test.npy")

    """!!!!!!!!!"""
    models_path = "D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/1_Paralelno/MNIST_Conv/"

    process = psutil.Process(os.getpid())


    def get_cpu_ram():
        cpu = psutil.cpu_percent(interval=0.1)
        ram = process.memory_info().rss / (1024 ** 2)  # MB
        return cpu, ram


    def useModel(x_test, y_test, modelPath):

        testModel = tf.keras.models.load_model(modelPath)
        res = []

        testModel.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        loss, acc = testModel.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(testModel.predict(x_test[:30]), axis=1)

        print(f"Model: {testModel.name}")
        print(f"Model loss: {loss}")
        print(f"Model accuracy: {acc}")
        print(f"Predicted results \n : {y_pred}")

        hit = 0
        for i in range(30):
            if y_pred[i] == y_test[i]:
                hit += 1

        predictedPerc = (hit / 30) * 100
        print(hit)
        print(f"Predicted to True acc: {predictedPerc}%")

        return acc, loss, predictedPerc

    res = []

    """Tree testing"""

    print("Tree testing\n")

    start = time.time()
    cpu_before, ram_before = get_cpu_ram()

    accuracy, usageloss, percentage = 0, 0, 0
    modelsNumber = 0
    bestAcc = 0

    for root, dirs, files in os.walk(models_path, topdown=False):
        # Check if this folder contains a model
        if "saved_model.pb" in files:
            print(f"\nTesting model in: {root}")

            try:
                modelsNumber += 1
                acc, loss, perc = useModel(x_test, y_test, root)
                accuracy += acc
                usageloss += loss
                percentage += perc

                if acc > bestAcc:
                    bestAcc = acc

            except Exception as e:
                print(f"Error loading {root}: {e}")

    end = time.time()
    cpu_after, ram_after = get_cpu_ram()

    print(f"\n Time per cycle: {end - start:.3f} sec")
    print(f"CPU and RAM usage: {cpu_after, ram_after}")

    print(f"Avg. Acc: {accuracy / modelsNumber}")
    print(f"Avg. Loss: {usageloss / modelsNumber}")
    print(f"Avg. Perc: {percentage / modelsNumber}")
    print(f"Best Model Acc: {bestAcc}")

    """One by One - One object
       Number of models for one object >
       Each models result is an input for the next model
       
       result = best acc, time spent, avg acc, avg loss, cpu, ram
    """

    def get_model_paths(base_dir):
        paths = []

        for root, dirs, files in os.walk(base_dir):
            if "saved_model.pb" in files:
                paths.append(root)

        # Optional: sort by name
        paths.sort()

        return paths

    def sequential_improvement_test(model_paths, X_test, y_test):
        best_accuracy = None
        best_model_path = None
        accuracy, usageloss = 0, 0
        modelsNumber = 0
        predictedPerc = 0

        for path in model_paths:
            print(f"\nTesting: {path}")

            try:
                model = tf.keras.models.load_model(path, compile=False)
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )

                loss, acc = model.evaluate(X_test, y_test, verbose=0)
                y_pred = np.argmax(model.predict(x_test[:30]), axis=1)

                print(f"Accuracy: {acc:.4f}")

                # First model → always accepted
                if best_accuracy is None:
                    hit = 0
                    for i in range(30):
                        if y_pred[i] == y_test[i]:
                            hit += 1

                    predictedPerc += (hit / 30) * 100
                    
                    best_accuracy = acc
                    best_model_path = path
                    print("Baseline model set")
                    modelsNumber += 1

                else:
                    if acc > best_accuracy:
                        print("Improvement! Keeping model")
                        best_accuracy = acc
                        best_model_path = path
                        accuracy += acc
                        usageloss += loss

                    else:
                        print("Worse than previous. STOPPING.")
                        break

                del model

            except Exception as e:
                print(f"Error: {e}")

        return {
            "best_model": best_model_path,
            "accuracy": float(best_accuracy),
            "models_number": modelsNumber,
            "avgaccuracy": accuracy,
            "usageloss": usageloss,
            "percentage": predictedPerc

        }


    """Sequential testing"""

    model_paths = get_model_paths("D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/1_Paralelno/MNIST_Conv/")

    start = time.time()
    cpu_before, ram_before = get_cpu_ram()

    result = sequential_improvement_test(
        model_paths,
        x_test,
        y_test,
    )

    end = time.time()
    cpu_after, ram_after = get_cpu_ram()

    print("\nBest model:", result["best_model"])
    print("Accuracy:", result["accuracy"])

    print(f"\n Time per cycle: {end - start:.3f} sec")
    print(f"CPU and RAM usage: {cpu_after, ram_after}")

    print(f"Avg. Acc: {result['avgaccuracy'] / result['models_number']}")
    print(f"Avg. Loss: {result['usageloss'] / result['models_number']}")
    print(f"Avg. Perc: {result['percentage'] / result['models_number']}")
    print(f"Best Model Acc: {bestAcc}")

    """Free structure - Random way of saving models
    
       result = best acc, time spent, avg acc, avg loss, cpu, ram
    """

    process = psutil.Process(os.getpid())
    model_paths = get_model_paths("D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/3_Mixed")

    """Mixed testing - Models one by one"""

    print("Mixed\n")

    start = time.time()
    cpu_before, ram_before = get_cpu_ram()

    accuracy, usageloss, percentage = 0, 0, 0
    modelsNumber = 0
    bestAcc = 0

    for folder_name in os.listdir(models_path):
        folder_path = os.path.join(models_path, folder_name)
        # Only folders
        if not os.path.isdir(folder_path):
            continue

        # Check if it's a TensorFlow SavedModel
        model_file = os.path.join(folder_path, "saved_model.pb")

        if not os.path.isfile(model_file):
            print(f"Skipped (no model): {folder_name}")
            continue

        try:
            modelsNumber += 1
            acc, loss, perc = useModel(x_test, y_test, folder_path)
            accuracy += acc
            usageloss += loss
            percentage += perc

            if acc > bestAcc:
                bestAcc = acc

        except Exception as e:
            print(f"Error loading {folder_name}: {e}")

    end = time.time()
    cpu_after, ram_after = get_cpu_ram()

    print(f"\n Time per cycle: {end - start:.3f} sec")
    print(f"CPU and RAM usage: {cpu_after, ram_after}")

    print(f"Avg. Acc: {accuracy / modelsNumber}")
    print(f"Avg. Loss: {usageloss / modelsNumber}")
    print(f"Avg. Perc: {percentage / modelsNumber}")
    print(f"Best Model Acc: {bestAcc}")

    """ Multi-process processing - simultanious testing of multiple modesl via multithreading
    Paralel,
    Tree,
    Free
    """


# if __name__ == '__main__':
#
#     class TestStrategy(ABC):
#         @abstractmethod
#         def test(self, models, test_data):
#             pass
#
#
#     def load_savedmodels(base_dir):
#         models = []
#
#         for folder_name in os.listdir(base_dir):
#             folder_path = os.path.join(base_dir, folder_name)
#
#             # Only folders
#             if not os.path.isdir(folder_path):
#                 continue
#
#             # Check if it's a TensorFlow SavedModel
#             model_file = os.path.join(folder_path, "saved_model.pb")
#
#             if not os.path.isfile(model_file):
#                 print(f"Skipped (no model): {folder_name}")
#                 continue
#
#             try:
#                 model = tf.keras.models.load_model(folder_path, compile=False)
#                 models.append((model, folder_name))
#                 print(f"Loaded: {folder_name}")
#
#             except Exception as e:
#                 print(f"Error loading {folder_name}: {e}")
#
#         return models
#
#
#     def load_torch_models(path):
#         model = torch.load(path)
#         model.eval()
#         return model
#
#
#     class OneByOneTest(TestStrategy):
#         def test(self, models, test_data):
#             results = []
#
#             for model in models:
#                 loss, acc = model.evaluate(test_data, verbose=0)
#                 results.append({
#                     "model": model.name,
#                     "loss": loss,
#                     "accuracy": acc
#                 })
#
#             return results
#
#
#     class CompareAllTest(TestStrategy):
#         def test(self, models, test_data):
#             return {
#                 model.name: model.evaluate(test_data, verbose=0)
#                 for model in models
#             }
#
#
#     class ThresholdTest(TestStrategy):
#         def __init__(self, min_accuracy):
#             self.min_accuracy = min_accuracy
#
#         def test(self, models, x_test):
#             passed = []
#             failed = []
#
#             for model in models:
#                 loss, acc = model.evaluate(x_test, y_test, verbose=0)
#                 preds = model.predict(x_test[:20])
#                 y_pred = preds.argmax(axis=1)
#
#                 if acc >= self.min_accuracy:
#                     passed.append(model.name)
#                 else:
#                     failed.append(model.name)
#
#             return {
#                 "passed": passed,
#                 "failed": failed
#             }
#
#
#     class ModelTestingApp:
#         def __init__(self, models_path, test_strategy):
#             self.models = load_savedmodels(models_path)
#             self.strategy = test_strategy
#
#         def run(self, test_data):
#             if not self.models:
#                 raise RuntimeError("No models found!")
#             return self.strategy.test(self.models, test_data)
#
#
#     models_path = "D:/PythonProjects/MagRobota/Stattya_SavedModelsForTesting/1_Paralelno/MNIST_Conv/"
#
#
#     # Choose strategy
#     strategy = ThresholdTest(min_accuracy=0.85)
#
#     app = ModelTestingApp(models_path, strategy)
#
#     #TESTING DATASET
#     # test_dataset = [1, 2, 3, 4]
#     test_dataset = np.load("D:/PythonProjects/MagRobota/processed_dataset/X_test.npy")
#     y_test = np.load("D:/PythonProjects/MagRobota/processed_dataset/y_test.npy")
#
#     results = app.run(test_dataset)
#     print(results)
#
#     # ADD RUNTIME TO EACH TEST