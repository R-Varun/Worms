import numpy as np
import sklearn
import sklearn.ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
import skvideo.io
from skimage import transform, filters, feature, color, measure, morphology, util, io, draw, exposure
import os
import sklearn.svm
import copy
from main import *


def trainWorm():
    classifier = Classifier()
    worm =  loadDataSet("train/wurm", asPythonArr=True)
    not_worm = loadDataSet("train/not_wurm", asPythonArr=True)
    dataSet = worm + not_worm
    dataSet = np.asarray(dataSet)
    labels = ["wurm" for i in worm] + ["not_wurm" for i in not_worm]
    classifier.train(dataSet, labels)
    # classifier.crossValidate(dataSet, labels)
    # TEST WORM CLASSIFIER
    test_worm = loadDataSet("test/wurm", asPythonArr=True)
    test_not_worm = loadDataSet("test/not_wurm", asPythonArr=True)

    test_data = test_worm + test_not_worm
    test_data = np.asarray(test_data)

    test_labels = ["wurm" for i in test_worm] + ["not_wurm" for i in test_not_worm]
    classifier.test(test_data, test_labels)
    print("end train")
    return classifier

trainWorm()