from skimage import transform, io, color
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


def loadDataSet(file, asPythonArr = True, size=(300,300)):
    dirs = os.listdir(file)
    # assume images
    dataset = []
    for image in dirs:
        if image.endswith(".png"):
            image_path = os.path.join(file, image)
            dataset.append(transform.resize(io.imread(image_path), size))


    if asPythonArr:
        return dataset
    return np.asarray(dataset)


ds = loadDataSet("not_wurm")

counter = 0
for image in ds:
    i = color.rgb2gray(image)
    for x in range(0, 180, 20):
        i = transform.rotate(i, x, cval=0)
        i = color.gray2rgb(i)
        io.imsave("./new_not_worm/" + str(counter) + ".png", i)
        counter += 1



