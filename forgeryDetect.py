from skimage import data, io, filters

image = data.coins()
lf = filters.gaussian(image, sigma = 2)
io.imshow(lf)
io.show()
