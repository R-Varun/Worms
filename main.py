import argparse
import copy
import pickle

import sklearn
import sklearn.ensemble
import sklearn.svm
import skvideo.io
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from convnet import *
from util.worm import *


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




class Classifier:
    def __init__(self, name="model"):
        # self.classifier = sklearn.svm.SVC(kernel="poly", degree=2, gamma='auto', max_iter=-1, verbose=True)
        # self.classifier = sklearn.ensemble.AdaBoostClassifier(sklearn.svm.SVC(probability=True, kernel="linear", max_iter= 100, C=100, verbose=True))
        # self.classifier = sklearn.svm.SVC(kernel="poly", degree=1, C=1, gamma='auto', max_iter=-1, verbose=True)
        # self.classifier = sklearn.svm.LinearSVC()
        # self.classifier = clf
        self.classifier = sklearn.svm.SVC(kernel="rbf", max_iter=100, C=100, verbose=True)
        self.trainSet = None
        self.name = name

    def load(self):
        self.classifier = pickle.load(open(self.name, 'rb'))

    def train(self, dataSet, labels, load=False):
        print("EXTRACTING FEATURES")
        ds = None
        if load:
            print("LOADING FEATURES")
            ds = np.load("feature_extracted.npy")
        else:
            ds = self.extract_features(dataSet)
            print("STORING FEATURES")

            np.save("feature_extracted", ds)

        print("TRAINING ON DATA")
        self.classifier.fit(ds, labels)

        # p = self.predict(dataSet)
        # print(metrics.accuracy_score(p, labels))
        # print(metrics.f1_score(p, labels, average='micro'))
        pickle.dump(self.classifier, open(self.name, 'wb'))

    def test(self, dataSet, labels):
        p = self.predict(dataSet)
        f1 = metrics.f1_score(p, labels, average='micro')
        acc = metrics.accuracy_score(p, labels)
        conf = metrics.confusion_matrix(p, labels)

        specs = {"F1":f1, "Accuracy":acc, "matrix": conf}
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

            mod = transform.resize(i, (300,300))
            f, im = feature.hog(mod, orientations=12, pixels_per_cell=(20, 20), cells_per_block=(15, 15), transform_sqrt=True,visualise=True, feature_vector=True, block_norm="L2-Hys")

            # mod = transform.resize(i, (100, 100))
            # f, im = feature.hog(mod, orientations=12, pixels_per_cell=(20, 20), cells_per_block=(15, 15),
            #                     transform_sqrt=True, visualise=True, feature_vector=True, block_norm="L2-Hys")

            # plt.imshow(im,cmap="gray")
            # plt.show()

            # f, im = feature.hog(mod, orientations=6, pixels_per_cell=(5, 5), cells_per_block=(4, 4), feature_vector=True,
            #                 block_norm='L2-Hys', visualise=True)

            # plt.imshow(im,cmap="gray")
            # plt.show()
            newDataSet.append(f)
        newDataSet = np.asarray(newDataSet)
        return newDataSet

    def crossValidate(self, dataset, labels):
        parameters = {'kernel': ('rbf',), 'C': [1, 10, 100, 50], 'max_iter': [-1, 10, 20, 30, 40,50]}
        clf = GridSearchCV(sklearn.svm.SVC(), parameters)


        ds = self.extract_features(dataset)

        clf.fit(ds, labels)
        order = clf.best_params_
        print(order)



def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
		for x in range(0, image.shape[1]- windowSize[1]+ 1, stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def sliding_window_pyramid(image, stepSize, windowSizePyramid):
    for windowSize in windowSizePyramid:
        # slide a window across the image
        for y in range(0, image.shape[0] - windowSize[0] + 1, stepSize):
            for x in range(0, image.shape[1]- windowSize[1]+ 1, stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])




shouldTrainWorm = False
shouldTrainChamber = False


def trainWorm():
    classifier = Classifier()
    worm =  loadDataSet("new_worm", asPythonArr=True)
    not_worm = loadDataSet("new_not_worm", asPythonArr=True)
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


def analyze(input_dir, output_dir="./", chamber_type=False, skip_frames = 1):
    videodata = skvideo.io.vreader(input_dir)
    prevFrame = None
    track = 0


    # LOAD WORM CLASSIFIER

    classifier = Classifier()
    if shouldTrainWorm:
        classifier = trainWorm()
    classifier.load()


    chamber_classifier = Classifier(name="chambers")
    if shouldTrainChamber:
        chamber = loadDataSet("chamber", asPythonArr=True)
        chamber_classifier.classifier = sklearn.svm.LinearSVC()
        not_chamber = loadDataSet("not_chamber", asPythonArr=True)
        dataset = np.asarray(chamber + not_chamber)
        labels = ["chamber" for i in chamber] + ["not_chamber" for i in not_chamber]
        # chamber_classifier.crossValidate(dataset, labels)
        chamber_classifier.train(dataset, labels)

    print("DONE TRAINING")

    # return

    ww = WormWrangler()

    wcr = WormChamberWrangler()
    frame_count = 0
    for frame in videodata:
        # TO SKIP FRAMES
        frame_count += 1
        if not (frame_count % skip_frames == 0):
            continue

        #END SKIP FRAMES


        mod = frame
        # KEEP A COPY IN CASE MODIFIED
        drawVer = copy.deepcopy(mod)


        mod = color.rgb2grey(mod)
        thresh = filters.threshold_otsu(mod)
        # bin = mod > thresh
        # plt.imshow(binary)
        # plt.show()

        if frame_count == skip_frames and chamber_type:
            circles = findChambers(mod)
            for circle in circles:
                perimeter =  circle_perimeter(circle[0], circle[1], circle[2])
                wcr.add((circle[0], circle[1]), circle[2])
                try:
                    drawVer[perimeter[1], perimeter[0]] = (0,255,20)
                except:
                    print("out of range")



        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        contours = measure.find_contours(mod, .95)

        for contour in contours:
            lx = int(min(contour[:, 1]))
            hx = int(max(contour[:, 1]))

            ly = int(min(contour[:, 0]))
            hy = int(max(contour[:, 0]))

            dx = hx - lx
            dy = hy - ly
            # if dx <= 0 or dy <= 0:
            #     continue
            #
            if dx * dy < 50:
                continue
            if not wcr.isInsideChamber(((lx + hx) / 2, (ly + hy) / 2)) and chamber_type:
                continue
            nly = clamp(ly - 10, 0, len(mod))
            nhx = clamp(hx + 10, 0, len(mod[0]))
            nhy = clamp(hy + 10, 0, len(mod))
            nlx = clamp(lx - 10, 0, len(mod[0]))

            # plt.imshow(mod[nly:nhy, nlx:nhx])
            # plt.show()
            # ans = classifier.predictOne(mod[nly:nhy, nlx:nhx])
            ans = predict(mod[nly:nhy, nlx:nhx])
            # ans = "wurm"
            if ans == 1:
                bitmap = mod[nly:nhy, nlx:nhx]

                coord = (lx + hx) / 2, (ly + hy) / 2
                ww.stage(bitmap, coord)

                c = plt.Rectangle((lx -10, ly - 10), hx - lx + 10, hy - ly + 10, color='r', fill=False)
                ax.add_patch(c)

                rr, cc = draw.polygon_perimeter([ly, ly, hy, hy, ly], [lx, hx, hx, lx, lx ])
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


        ww.next()
        print("FRAME:", frame_count)



    # At the end, create the video
    ww.pruneCandidates()
    for i, candidate in enumerate(ww.candidates):
        vid = np.multiply(np.array(candidate.frames), 255)

        skvideo.io.vwrite("output-" + str(i) + ".mp4", vid)
        print("VIDEO WRITTEN")
        # writer.close()

def clamp(val, lowerbound, upperbound):
    val = min(val, upperbound)
    val = max(val, lowerbound)
    return val

def findChambers(img, sizemin=50, sizemax=70):
    i = feature.canny(img)
    # Detect two radii

    hough_radii = np.arange(sizemin, sizemax, 3)
    hough_res = hough_circle(i, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=300)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(img)
    buckets = []


    circles = []
    # matches overlapping circles so that only distant circles count independently
    for center_y, center_x, radius in zip(cy, cx, radii):
        foundMatch = False
        for i in buckets:
            this_x, this_y, this_cicle = i
            cur_dist = distance(this_x, this_y, center_x, center_y)
            if cur_dist < sizemin:
                foundMatch = True
                break
        if not foundMatch:
            buckets.append((center_x, center_y, circle_perimeter(center_y, center_x, radius)))
            circles.append((center_x, center_y, radius))

    return circles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some Worms.')
    parser.add_argument("-input", help="input video directory")
    parser.add_argument("-output", help="output video directory")
    parser.add_argument("--chamber", help="whether chambered or not")

    args = parser.parse_args()
    chamber = False
    if args.chamber:
        chamber = True


    print(args.input, chamber)
    analyze(args.input, output_dir=args.output, chamber_type=chamber)


