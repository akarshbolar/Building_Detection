import numpy as np
import os
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from scipy.misc import imsave
from scipy.cluster.vq import *
import csv
import cv2
from skimage.segmentation import slic
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_float
# Path to input images
raster = "out3.tiff"

input = gdal.Open(raster, gdal.GA_ReadOnly)
print input.RasterCount
# Loop through all raster bands
band_list = []
for b in range(1, input.RasterCount + 1):
    band = input.GetRasterBand(b)
    band_list.append(band.ReadAsArray())
print len(band_list[0][0])
print "Driver: {}/{}".format(input.GetDriver().ShortName,
                             input.GetDriver().LongName)
print "Size is {} x {} x {}".format(input.RasterXSize,
                                    input.RasterYSize,
                                    input.RasterCount)
print "Projection is {}".format(input.GetProjection())

transform = input.GetGeoTransform()
if transform:
    print "Origin = ({}, {})".format(transform[0], transform[3])
    print "Pixel Size = ({}, {})".format(transform[1], transform[5])


# Stack 2D arrays (image) into a single 3D array
stack = np.dstack(band_list)
print len(stack[0][0])
# Get the dimensions of stack
r, c, n = stack.shape
print r,c, n
# Total number of data samples in stack
n_samples = r * c

# Flatten stack to r
flat = stack.reshape((n_samples, n))
print len(flat)
print stack.shape[0], stack.shape[1]

# Apply k-means clustering
n_clusters = 4
centroids, variance = kmeans(flat.astype('double'), n_clusters)
print centroids

code, dist = vq(flat, centroids)
cluster_img = code.reshape(stack.shape[0], stack.shape[1])
print("hello")
# Save clustered image
imsave('kanteerava.tif', cluster_img)
