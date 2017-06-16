from skimage import data, io, filters, feature
from sklearn.metrics.pairwise import euclidean_distances

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

dist_Treshhold = 15
sim_Treshhold = 1000
#step 1: get img
image = data.coins()
for i in range(0, 50):
	for j in range(0, 50):
		image[i][j] = image[200+i][200+j]
io.imshow(image)
io.show()
#step 2: apply filter
lf_img = filters.gaussian(image, sigma = 2)
#step 3: get subimages
subimgs = seg_img(lf_img, 25)
#step 4: get features for subimages
print "step 4"
descriptor = LBP_Descriptor(24, 3)
lbp = {}
print len(subimgs.keys())
for i in subimgs.keys():
	lbp[i] = (descriptor.LBPcalc(subimgs[i])).flatten()

#step 5: euclidean distance to subimages
print "step 5"
elements = lbp.values()
distMatrix = euclidean_distances(elements, elements)
#step 6: Choose similar subimgs
minimum = []
for i in xrange(1, len(distMatrix)):
	minimum.append(min(distMatrix[i][:i]))
match_pos = []
print minimum
for i in range(0,len(minimum)):
	if minimum[i] <= 300:
		for key, value in lbp.iteritems():
			if (value.all() == elements[i].all()):
				match_pos.append(key)
print match_pos
#io.imshow(lbp)
#io.show()
