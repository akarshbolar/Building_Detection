%matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image



def ld_img(fname):
    data = mpimg.imread(fname)
    return data

def pd_img(data, filler):
    if len(data.shape) < 3:
        data = np.lib.pad(data, ((filler, filler), (filler, filler)), 'reflect')
    else:
        data = np.lib.pad(data, ((filler, filler), (filler, filler), (0,0)), 'reflect')
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def image_affix(img, groundtruth_img):
    nChannels = len(groundtruth_img.shape)
    w = groundtruth_img.shape[0]
    h = groundtruth_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, groundtruth_img), axis=1)
    else:
        groundtruth_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        groundtruth_img8 = img_float_to_uint8(groundtruth_img)          
        groundtruth_img_3c[:,:,0] = groundtruth_img8
        groundtruth_img_3c[:,:,1] = groundtruth_img8
        groundtruth_img_3c[:,:,2] = groundtruth_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, groundtruth_img_3c), axis=1)
return cimg

def img_crop(im, w, h, stride, ext):
    patch_list = []
    iwidth = im.shape[0]
    iheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if not is_2d:
        im = np.lib.pad(im, ((ext, ext), (ext, ext), (0,0)), 'reflect')
    for i in range(ext,iheight+ext,stride):
        for j in range(ext,iwidth+ext,stride):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                #im_patch = im[j:j+w, i:i+h, :]
                im_patch = im[j-ext:j+w+ext, i-ext:i+h+ext, :]
            patch_list.append(im_patch)
    return patch_list

# Extract features for a given image
def extract_img_features(filename, stride):
    img = ld_img(filename)
    img_patches = img_crop(img, p_s, p_s, stride, filler)
    X = np.asarray([img_patches[i] for i in range(len(img_patches))])
    return X



p_s = 16
w_s = 72
filler = (w_s - p_s) // 2
stride = 16



def label_to_img(iwidth, iheight, w, h, labels, stride):
    im = np.zeros([iwidth, iheight])
    idx = 0
    for i in range(0,iheight,stride):
        for j in range(0,iwidth,stride):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255
    
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



from cnn_model import CnnModel

from keras import backend; print(backend._BACKEND)
from keras import backend as K
import os
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
print ("Change Keras Backend to Theano")        
set_keras_backend("theano")  
from keras import backend; print(backend._BACKEND)
import keras.backend as K
K.set_image_dim_ordering('th')
# Load model from disk
model = CnnModel()
model.load('weights.h5')



image_number = 3 # This value can be changed

test_directory = "testing/test_"
directory = test_directory + str(image_number) + "/test_" + str(image_number) + ".png"
Xi = extract_img_features(directory, stride)
print(1)

if K.image_dim_ordering() == 'th':
    print("igfv")
    Xi = np.rollaxis(Xi, 3, 1)
print(2)
Zi = model.model.predict(Xi)
Zi = Zi[:,0] > Zi[:,1]

img_ = ld_img(directory)
w = img_.shape[0]
h = img_.shape[1]

predicted_im = label_to_img(w, h, p_s, p_s, Zi, stride)
cimg = image_affix(img_, predicted_im)
fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
plt.imshow(cimg, cmap='Greys_r')

new_img = make_img_overlay(img_, predicted_im)
plt.imshow(new_img)
