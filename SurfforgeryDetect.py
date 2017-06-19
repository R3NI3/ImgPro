from skimage import data, io, filters, feature
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from math import sqrt
import numpy as np
import cv2

#********************* Algorithm *********************
# step 0: Initialization
surf = cv2.xfeatures2d.SURF_create(300, extended = True)

#step 1: get img OK
print "step 1"
image = data.coins()
for i in range(50, 100):
	for j in range(50, 100):
		image[i][j] = image[200+i][200+j]
#io.imshow(image)
#io.show()

#step 2: Compute surf OK
kp, desc = surf.detectAndCompute(image, None)
img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
#io.imshow(img2),io.show()

#step 4: Feature matching
tree = KDTree(desc)
result = map(lambda x:tree.query(x,3), desc)
result = filter(lambda res: (res[0][1]/res[0][2]) < 0.15, result)
kp_idx = []
for i in result:
	for j in i[1]:
		if j not in kp_idx:
			kp_idx.append(j)
print kp_idx
kp = map(lambda idx:kp[idx], kp_idx)
img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
io.imshow(img2),io.show()


