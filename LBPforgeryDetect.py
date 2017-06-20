from skimage import data, io, filters, feature
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from math import sqrt
import cv2
import numpy as np
import glob

class LBP_Descriptor:
	def __init__(self, numPoints, radius):
		self.numPoints = numPoints
		self.radius = radius
	
	def LBPcalc(self, subimage):
		lbp = feature.local_binary_pattern(subimage, self.numPoints, 
												self.radius, method = "uniform")
		return lbp					

def seg_img(image, radius):
	subimg = {}
	for x in range(0,(image.shape[0]-(2*radius)), 2*radius):
		for y in range(0,(image.shape[1]-(2*radius)), 2*radius):
			pos = (x, y)
			subimg[pos] = (image[x:x+(2*radius),y:y+(2*radius)])
			
	return subimg


#********************* Algorithm *********************
# step 0: Initialization
radius = 5 # sub-images radius 
subimg_step = 1 # num of pixels to be skiped for subimages
numpoints = 8 #number of points considered in feature extration LBP
LBP_radius = 1 # distace of the points to get
dist_Treshold = 15 #threshold for distance
sim_Treshold = 20 #dissimilarity threshold
lex_dist = 100

images = glob.glob('*.jpg')
for img in images:
	#step 1: get img OK
	#print "step 1"
	#image = data.coins()
	image = cv2.imread(img, 0)
	#for i in range(0, 50):
	#	for j in range(0, 50):
	#		image[i][j] = image[200+i][200+j]
	#io.imshow(image)
	#io.show()

	#step 2: apply filter OK
	#print "step 2"
	lf_img = filters.gaussian(image, sigma = 2)

	#step 3: get subimages OK
	#print "step 3"
	subimages,centers = zip(*[(image[center1-radius:center1+radius, 
									center2-radius:center2+radius],
									(center1,center2))
								for center1 in xrange(radius,
												image.shape[0]-radius,
												subimg_step)
								for center2 in xrange(radius,
												image.shape[1]-radius,
												subimg_step)])

	#step 4: get features for subimages OK
	#print "step 4"
	LBP_mat = list(map(lambda subimg:feature.local_binary_pattern(subimg,
													numpoints,
													LBP_radius,
													method = "uniform"),
						subimages))
	#transform matrices in rows list OK
	LBP_list = np.array(list(map(lambda subimg:subimg.flatten(), LBP_mat)))

	#lexicographic order maybe OK
	ind = np.lexsort(LBP_list.T[::-1])
	lex_order = np.array(list(LBP_list[ind]))
	new_centers = [centers[i] for i in ind]

	#step 5: euclidean distance to subimages OK
	#print "step 5"
	dist = [map(lambda lex:np.linalg.norm(lex_order[i] - lex_order[lex]) 
									if np.linalg.norm(np.array(new_centers[i]) - np.array(new_centers[lex])) > 10
									else 10000, 
									xrange(i+1, i + lex_dist))
									for i in xrange(0, (len(lex_order)-lex_dist))]

	min_dist = np.array(map(lambda d:min(d), dist))
	#print min_dist
	#step 6: Choose similar subimgs creating similar set OK
	#print "step 6" 
	cnt_list = np.argwhere(min_dist < sim_Treshold)
	pos_set = map(lambda cnt:new_centers[cnt[0]], cnt_list)

	#step 7: clear image OK
	#print "step 7"
	count = 0
	for i in xrange(1, image.shape[0]-1):
		for j in xrange(1, image.shape[1]-1):
			if image[i][j] == 0:
				if 0 in [image[i-1][j],image[i+1][j],image[i][j-1],image[i][j+1]]:
					count+=1

	for t in pos_set:
		image[t[0]][t[1]] = 0

	count2 = 0
	for i in xrange(1, image.shape[0]-1):
		for j in xrange(1, image.shape[1]-1):
			if image[i][j] == 0:
				if 0 in [image[i-1][j],image[i+1][j],image[i][j-1],image[i][j+1]]:
					count2+=1
	diff = count2 - count
	print diff
	if diff > 1000:
		print "Forged"
	else:
		print "Authentic"
	#io.imshow(image)
	#io.show()
