
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy

import tensorflow as tf
import pygad

from pygad import kerasga

print(tf.__version__)

# Not tested ----------------------------------------------------------------------------------------
# studyFirstTypeModels studyHighSoundTypeModels

def studyGeneticTypeModels(trainFeatures, trainLabels, testFeatures, testLabels, fileNames, savePath, headers):

    modelType = "GeneticAlgorithm"

    model_start = headers[2]
    model_start = model_start.split("Гц")[0]
    model_end = headers[-1]
    model_end = model_end.split("Гц")[0]

    # normalize = layers.Normalization()
    # normalize.adapt(trainFeatures)

    resultSet = []
    countSet = []

    x_train = numpy.reshape(trainFeatures, (trainFeatures.shape[0], trainFeatures.shape[1],1))
    x_test = numpy.reshape(testFeatures, (testFeatures.shape[0], testFeatures.shape[1],1))

    # data_inputs = trainFeatures
    data_inputs = trainFeatures
    data_outputs = trainLabels

    for i in range(1):
        # Model study
        (x_train.shape[1], 1)
        input_layer = tf.keras.layers.Input(334)
        dense_layer = tf.keras.layers.Dense(25, activation='relu')(input_layer)
        output_layer = tf.keras.layers.Dense(1, activation='softmax')(dense_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # model = tf.keras.Sequential([
        #     layers.LSTM(units=100, return_sequences=True, input_shape=(334, 1)),
        #     layers.Dropout(0.2),
        #     layers.LSTM(units=75, return_sequences=True),
        #     layers.Dropout(0.2),
        #     layers.LSTM(units=50),
        #     layers.Dropout(0.2),
        #     layers.Dense(101),
        # ])

        keras_ga = pygad.kerasga.KerasGA(model=model,
                                         num_solutions=10)

        def fitness_func(solution, sol_idx):
            # global data_inputs, data_outputs, keras_ga, model

            model_weights_matrix = kerasga.model_weights_as_matrix(model=model,
                                                                   weights_vector=solution)
            model.set_weights(weights=model_weights_matrix)

            predictions = model.predict(data_inputs)
            # predictions = numpy.argmax(model.predict(data_inputs), axis=1)

            bce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            # bce = tf.keras.losses.BinaryCrossentropy()
            error = bce(data_outputs, predictions).numpy()

            fitness = 1.0 / (bce(data_outputs, predictions).numpy() + 0.00000001)

            return fitness

        def on_genration(ga_instance):
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(ga_instance.last_generation_fitness)[1]),)

        ga_instance = pygad.GA(num_generations=100,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               initial_population=keras_ga.population_weights,
                               on_generation=on_genration,
                               suppress_warnings=True)

        ga_instance.run()

        fig = ga_instance.plot_result(title="Iteration vs Fitness", linewidth=4)

        solution, solution_fitness, _ = ga_instance.best_solution()
        print("Params of the best solution:\n{solution}".format(solution=solution), end="\n\n")
        print("Length of the best solution:", len(solution), end="\n\n")
        print("Fitness value of the best solution:\n{solution_fitness}".format(solution_fitness=solution_fitness), end="\n\n")

        best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                      weights_vector=solution)

        model.set_weights(best_solution_weights)
        predictions = model.predict(testFeatures)
        # predictions = numpy.argmax(model.predict(x_test), axis=1)

        print("Predictions: \n", predictions, end="\n\n")

        print("Correct_outputs: \n", data_inputs, end="\n\n")


        # y_pred = numpy.argmax(sound_model.predict(x_test), axis=1)
        # print(y_pred)
        #
        # y_true = testLabels
        #
        # rounded = []
        # for v in y_pred:
        #     if v > 60:
        #         rounded.append(100)
        #     else:
        #         rounded.append(0)
        #
        # test_acc = sum(rounded == y_true) / len(y_true)
        # print(f'Test set accuracy: {test_acc:.0%}')
        #
        # count = countGuesses(fileNames, y_true, y_pred)
        #
        # # Model check
        # if test_acc > 0.80:
        #     path = "%s/%s, %f" % (savePath, modelType, test_acc)
        #     saveModel(sound_model, path)
        #
        #     with open(path + '\params.txt', 'w') as f:
        #         f.write(model_start + "\n")
        #         f.write(model_end)
        #
        # resultSet.append(test_acc)
        # countSet.append(count)

    # return countSet, resultSet
    # 5 models