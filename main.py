

import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import skvideo.io
from skimage import transform, filters, feature, color, measure, morphology, util, io, draw
import os
import sklearn.svm
import copy
from worm import *

import pickle


from skimage.morphology import square
# import skimage.transform
# import skvideo.datasets


def loadDataSet(file, asPythonArr = True, size=(100,100)):
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




class Classifier:
    def __init__(self, name="model"):
        self.classifier = sklearn.svm.SVC(kernel="poly", degree=1, gamma='auto', max_iter=-1, verbose=True)
        self.trainSet = None
        self.name = name

    def load(self):
        self.classifier = pickle.load(open(self.name, 'rb'))

    def train(self, dataSet, labels):
        ds = self.extract_features(dataSet)
        self.classifier.fit(ds, labels)
        p = self.predict(dataSet)
        # print(metrics.accuracy_score(p, labels))
        # print(metrics.f1_score(p, labels, average='micro'))
        pickle.dump(self.classifier, open(self.name, 'wb'))

    def test(self, dataSet, labels):
        p = self.predict(dataSet)
        f1 = metrics.f1_score(p, labels, average='micro')
        acc = metrics.accuracy_score(p, labels)

        specs = {"F1":f1, "Accuracy":acc}
        print(specs)
        return specs



    def predict(self,  pred):
        return self.classifier.predict(self.extract_features(pred))

    def predictOne(self, pred):
        pred = self.extract_features([pred])
        pred.reshape(-1, 1)
        return self.classifier.predict(pred)[0]

    def extract_features(self, dataSet):
        newDataSet = []
        for i in dataSet:
            i = color.rgb2gray(i)
            i = filters.sobel(i)

            mod = transform.resize(i, (200,200))
            f, im = feature.hog(mod, orientations=12, pixels_per_cell=(20, 20), cells_per_block=(5, 5), transform_sqrt=True,visualise=True, feature_vector=True, block_norm="L2-Hys")

            # plt.imshow(im,cmap="gray")
            # plt.show()

            # f, im = feature.hog(mod, orientations=6, pixels_per_cell=(5, 5), cells_per_block=(4, 4), feature_vector=True,
            #                 block_norm='L2-Hys', visualise=True)

            # plt.imshow(im,cmap="gray")
            # plt.show()
            newDataSet.append(f)
        newDataSet = np.asarray(newDataSet)
        return newDataSet



def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
		for x in range(0, image.shape[1]- windowSize[1]+ 1, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])





def analyze():
    videodata = skvideo.io.vreader("9.avi")
    prevFrame = None
    track = 0


    # classifier load
    classifier = Classifier()
    # worm = loadDataSet("wurm", asPythonArr=True)
    # not_worm = loadDataSet("not_wurm", asPythonArr=True)
    #
    #
    # dataSet = worm + not_worm
    # dataSet = np.asarray(dataSet)
    # labels = ["wurm" for i in worm] + ["not_wurm" for i in not_worm]
    #
    # classifier.train(dataSet, labels)
    #
    #
    # test_worm = loadDataSet("test/wurm", asPythonArr=True)
    # test_not_worm = loadDataSet("test/not_wurm", asPythonArr=True)
    #
    # test_data = test_worm + test_not_worm
    # test_data = np.asarray(test_data)
    # test_labels = ["wurm" for i in test_worm] + ["not_wurm" for i in test_not_worm]
    # classifier.test(test_data, test_labels)
    classifier.load()




    print("DONE TRAINING")

    ww = WormWrangler()


    frame_count = 0
    for frame in videodata:
        frame_count += 1
        if not (frame_count % 20 == 0):
            continue

        mod = transform.resize(frame, (1000,1000))
        drawVer = copy.deepcopy(mod)

        mod = color.rgb2grey(mod)


        slides = sliding_window(mod, 50, (100,100))

        fig, ax = plt.subplots()


        contours = measure.find_contours(mod, .8)

        for contour in contours:
            # print(contour)
            lx = int(min(contour[:, 1]))
            hx = int(max(contour[:, 1]))

            ly = int(min(contour[:, 0]))
            hy = int(max(contour[:, 0]))

            dx = hx - lx
            dy = hy - ly
            if dx <= 0 or dy <= 0:
                continue

            if dx * dy < 50:
                continue


            ans = classifier.predictOne(mod[ly - 10:hy + 10, lx- 10:hx + 10])

            if ans == "wurm":
                bitmap = mod[ly - 10:hy + 10, lx - 10:hx + 10]
                coord = (lx + hx) / 2, (ly + hy) / 2
                ww.stage(bitmap, coord)

                c = plt.Rectangle((lx -10, ly - 10), hx - lx + 10, hy - ly + 10, color='r', fill=False)
                ax.add_patch(c)
                # plt.imshow(mod[ly - 10:hy + 10, lx- 10:hx + 10])
                # plt.show()
                rr, cc = draw.polygon_perimeter([ly, ly, hy, hy, ly], [lx, hx, hx, lx, lx ])
                # drawVer[rr, cc] = [0,1,0]
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ww.next()

        print(drawVer.shape)
        # plt.imshow(drawVer)
        # plt.show()

        prevFrame = frame

    print(ww.candidates[0].coords)

    # writer = skvideo.io.FFmpegWriter("outputvid.mp4", outputdict={
    #     '-vcodec': 'libx264', '-b': '300000000'
    # })
    # f =  np.array(ww.candidates[0].frames)
    # for i in f:
    #     writer.writeFrame(i)
    vid = np.multiply(np.array(ww.candidates[0].frames), 255)
    #
    # writer = skvideo.io.FFmpegWriter("output.mp4", outputdict={
    #     '-vcodec': 'libx264', '-b': '300000000'
    # })
    #
    # for i in vid:
    #     writer.writeFrame(i)
    # writer.close()

    skvideo.io.vwrite("outputvid.mp4", vid)
    # writer.close()


analyze()
#
# videodata = skvideo.io.vreader("outputvid.mp4")
# for frame in videodata:
#     plt.imshow(frame)
#     plt.show()

