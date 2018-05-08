import numpy as np
from skimage import transform, filters, feature, color, measure, morphology, util, io, draw
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return ((y2-y1)**2 + (x2 - x1)**2)**.5


def distance2(coord1, coord2):
    return distance(coord1[0], coord1[1], coord2[0], coord2[1])


class WormCandidate:
    def __init__(self, bitmap, coords, resolution=(480, 680)):
        self.resolution = resolution
        self.coords = []
        self.coords.append(coords)

        self.frames = []
        self.frames.append(self.processBitmap(bitmap))



        self.x, self.y = coords


    def setResolution(self, resolution):
        if len(resolution) != 2:
            raise ValueError("Invalid resolution passed")

        self.resolution = resolution

    def processBitmap(self, bitmap):
        resized = transform.resize(bitmap, self.resolution)
        return color.gray2rgb(resized)
        try:
            resized = transform.resize(bitmap, self.resolution)
            return color.gray2rgb(resized)
        except:
            print(bitmap.shape)
            print(self.coords[-1])
            plt.imshow(bitmap)
            plt.show()


    def addFrame(self, bitmap, coords):
        self.frames.append(self.processBitmap(bitmap))
        self.coords.append(coords)
        self.x, self.y = coords


    def index_of_match_candidate(self, candidate):
        #TODO: implement based on pose in addition to distance
        s1 = self.x, self.y
        s2 = candidate.x, candidate.y

        index = distance(s1[0], s1[1], s2[0], s2[1])
        return index

    def index_of_match_coords(self, coords):
        #TODO: implement based on pose in addition to distance
        s1 = self.x, self.y
        s2 = coords

        index = distance(s1[0], s1[1], s2[0], s2[1])
        return index


class WormWrangler:
    def __init__(self, closenessIndex = 100):
        # array of longer-lasting worm-candidates
        self.candidates = []

        # array of guesses that could become new worm-candidates or part of another worm candidate
        self.frame_candidates = []

    def stage(self, bitmap, coords):
        self.frame_candidates.append((bitmap, coords))

    def next(self):
        if len(self.candidates) == 0:
            for i in self.frame_candidates:
                self.candidates.append(WormCandidate(i[0], i[1]))
        else:
            matchList = list(self.frame_candidates)
            for i in self.candidates:
                if len(matchList) == 0:
                    pass
                else:
                    best_index, best_match = min(enumerate(matchList), key = lambda x: i.index_of_match_coords(x[1][1]))
                    i.addFrame(best_match[0], best_match[1])
                    matchList.pop(best_index)

            for i in matchList:
                self.candidates.append(WormCandidate(i[0], i[1]))


        self.frame_candidates = []

    def pruneCandidates(self):

        keep = []
        for i in range(len(self.candidates)):
            candidate = self.candidates[i]
            movement = 0
            st = candidate.coords[0]
            for coord in candidate.coords[1:]:
                movement += distance2(st, coord)
                st = coord
            if movement > 50:
                keep.append(candidate)

        self.candidates = keep










class WormChamber:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def isInside(self, coord):
        return distance(coord[0], coord[1], self.center[0], self.center[1]) <= self.radius - 5



class WormChamberWrangler:
    def __init__(self, chambers=[]):
        self.chambers = chambers

    def isInsideChamber(self, coord):
        for i in self.chambers:
            if i.isInside(coord):
                return True
        return False
    def add(self, center, radius):
        self.chambers.append(WormChamber(center, radius))

















