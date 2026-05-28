# Write to Excel, Graphs, Numpy
import pathlib
import os
import xlsxwriter as xls
import matplotlib.pyplot as plt
import numpy as np
import keras.utils as img
import tensorflow as tf
# Pillow for processing images
from PIL import Image
#Unfreeze
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    """
    
    # https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/
    # Via Pillow
    
    """
    image = Image.open('D:\PythonProjects\MagRobota\App\Sample.png')

    # summarize some details about the image
    print(image.format)
    print(image.size)
    print(image.mode)

    numpydata = np.asarray(image)

    # <class 'numpy.ndarray'>
    print(type(numpydata))
    # Shape
    print(numpydata.shape)
    # Pixel by pixel values in RGB (Probably)
    print(numpydata)

    # Pillow image from numpyarray
    pilImage = Image.fromarray(numpydata)

    print(type(pilImage))
    print(pilImage.mode)
    print(pilImage.size)

    # Via Keras
    image = img.load_img('D:\PythonProjects\MagRobota\App\Sample.png')

    # Details about the image
    print(image.format)
    print(image.mode)
    print(image.size)

    # Convert image into  numpy array
    numpydata = img.img_to_array(image)
    print("Image is converted and NumPy array information :")

    # <class 'numpy.ndarray'>
    print(type(numpydata))
    # Shape
    print(numpydata.shape)
    # Pixel by pixel values in RGB (Probably)
    print(numpydata)

    # convert back to image
    image = img.array_to_img(numpydata)

    # <class 'PIL.PngImagePlugin.PngImageFile'>
    print("Converting NumPy array into image:", type(image))

    """
    
    https://www.tensorflow.org/tutorials/load_data/images
    By image
    
    """

    # Get data set and set directory
    datasetUrl = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    dataDir = tf.keras.utils.get_file(origin=datasetUrl,
                                       fname='flower_photos',
                                       untar=True)
    dataDir = pathlib.Path(dataDir)

    # Info about set
    imageCount = len(list(dataDir.glob('*/*.jpg')))
    print(imageCount)

    # Initial image data
    batchSize = 32
    imgHeight = 180
    imgWidth = 180

    # Split data to training and testing set
    train = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(imgHeight, imgWidth), batch_size=batchSize)

    test = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(imgHeight, imgWidth), batch_size=batchSize)

    classNames = train.class_names
    print(classNames)

    # Visualise
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classNames[labels[i]])
            plt.axis("off")
    plt.show()

    for imageData, labels in train:
        print(imageData.shape)
        print(labels.shape)
        break

        # image data = image number, wight, height, and RGB encoding
        # labels is label for each of 32 photos
        # 32 is Tensor size set as bathcSize at top

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Before trining we must convert RGB [0..255] to rabge of [0..1]
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    # AUTOTUNE = tf.data.AUTOTUNE
    #
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



    # Data conveting by hand

    # Get and sguffle data from path
    listDs = tf.data.Dataset.list_files(str(dataDir / '*/*'), shuffle=False)
    listDs = listDs.shuffle(imageCount, reshuffle_each_iteration=False)

    # Get class names
    classNames = np.array(sorted([item.name for item in dataDir.glob('*') if item.name != "LICENSE.txt"]))
    print(classNames)

    # Split set into Train and Test sets
    size = int(imageCount * 0.2)
    train = listDs.skip(size)
    test = listDs.take(size)

    # Initial image data
    batchSize = 32
    imgHeight = 180
    imgWidth = 180

    # Convert path to file into pair [lable, image]
    def getLabel(filePath):
        # Convert the path to a list of path components
        parts = tf.strings.split(filePath, os.path.sep)
        # The second to last is the class-directory
        classDir = parts[-2] == classNames
        # Integer encode the label
        return tf.argmax(classDir)

    def decodeImg(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [imgHeight, imgWidth])

    def processPath(filePath):
        label = getLabel(filePath)
        # Load the raw data from the file as a string
        img = tf.io.read_file(filePath)
        img = decodeImg(img)
        return img, label


    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train = train.map(processPath, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(processPath, num_parallel_calls=tf.data.AUTOTUNE)

    for image, label in train.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    # Performance
    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batchSize)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train = configure_for_performance(train)
    test = configure_for_performance(test)

    # Visualise
    imageBatch, labelBatch = next(iter(train))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imageBatch[i].numpy().astype("uint8"))
        label = labelBatch[i]
        plt.title(classNames[label])
        plt.axis("off")

    plt.show()
