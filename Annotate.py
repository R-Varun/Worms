import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from skimage import transform, filters, feature, color, measure, morphology, util, io, draw, exposure

from worm import *



videodata = skvideo.io.vreader("8.avi")

for frame in videodata:
	mod = color.rgb2grey(mod)


