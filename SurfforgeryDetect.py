from skimage import data, io, filters, feature
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from math import sqrt
import random
import numpy as np
import cv2
import glob

#********************* Algorithm *********************
# step 0: Initialization
surf = cv2.xfeatures2d.SURF_create(400, extended = True)
images = glob.glob('*.jpg')
for img in images:
	#step 1: get img OK
	#print "step 1"
	#image = data.coins()
	image = cv2.imread(img, 0)
	#for i in range(0, 50):
	#	for j in range(0, 50):
	#		image[i][j] = image[200+i][200+j]
	#io.imshow(image),io.show()
	#step 2: Compute surf OK
	#print "step 2"
	kp, desc = surf.detectAndCompute(image, None)
	#img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
	#io.imshow(img2),io.show()
	#step 3: Feature matching
	#print "step 3"
	tree = KDTree(desc)
	result = map(lambda x:tree.query(x,3), desc)
	#print "step 3.1"
	result = filter(lambda res: (res[0][1]/res[0][2]) < 0.35, result)
	#print "step 3.2"
	kp_idx = []
	for i in result:
		if np.linalg.norm(np.array(kp[i[1][0]].pt) - np.array(kp[i[1][1]].pt)) > 10:
			kp_idx.append(i[1][0])
			kp_idx.append(i[1][1])

	kp = map(lambda idx:kp[idx], kp_idx)
	#for k in kp:
	#	print k.pt
	img2 = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
	io.imshow(img2),io.show()

	#print len(kp)
	if len(kp)>1:
		print "forged"
	else:
		print "original"


