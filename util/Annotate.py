import skvideo.io

from util.worm import *



videodata = skvideo.io.vreader("8.avi")

for frame in videodata:
	mod = color.rgb2grey(mod)


