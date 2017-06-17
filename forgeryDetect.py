from skimage import data, io, filters, feature
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from math import sqrt
import numpy as np

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
sim_Treshold = 1000 #dissimilarity threshold
lex_dist = 100

#step 1: get img OK
print "step 1"
image = data.coins()
for i in range(0, 50):
	for j in range(0, 50):
		image[i][j] = image[200+i][200+j]
#io.imshow(image)
#io.show()

#step 2: apply filter OK
print "step 2"
lf_img = filters.gaussian(image, sigma = 2)

#step 3: get subimages OK
print "step 3"
subimages,centers = zip(*[(image[center1-radius:center1+radius, center2-radius:center2+radius],(center1,center2))
								for center1 in xrange(radius,
												image.shape[0]-radius,
												subimg_step)
								for center2 in xrange(radius,
												image.shape[1]-radius,
												subimg_step)])

#step 4: get features for subimages OK
print "step 4"
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
print "step 5"
dist = [map(lambda lex:np.linalg.norm(lex_order[i] - lex), 
										lex_order[i+1:i + lex_dist]) 
								for i in xrange(0, (len(lex_order)-lex_dist))]
min_dist = map(lambda d:min(d),dist)
print len(dist)
print min_dist[67669]
#print map(lambda x:min(x), dist)
#print dist
#print min(dist)
#print dist.index(min(dist))
#tree = KDTree(LBP_list)
#dist, ind = tree.query(LBP_list[:], k=2)
#pairs = tree.query_pairs(10)
#step 6: Choose similar subimgs creating similar set
print "step 6"
#io.imshow(lbp)
#io.show()
