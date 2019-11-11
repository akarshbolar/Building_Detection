import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import re
def ld_img(fname):
    return mpimg.imread(fname)

def pd_img(data, filler):
    if len(data.shape) < 3:
        data = np.lib.pad(data, ((filler, filler), (filler, filler)), 'reflect')
    else:
        data = np.lib.pad(data, ((filler, filler), (filler, filler), (0,0)), 'reflect')
    return data
    
def crop_groundtruth(im, w, h, stride):
    assert len(im.shape) == 2, 'Not GreyScale.'
    patch_list = []
    iwidth = im.shape[0]
    iheight = im.shape[1]
    for i in range(0,iheight,stride):
        for j in range(0,iwidth,stride):
            im_patch = im[j:j+w, i:i+h]
            patch_list.append(im_patch)
    return patch_list
    
def img_crop(im, w, h, stride, filler):
    assert len(im.shape) == 3, 'Please use RBG image.'
    patch_list = []
    iwidth = im.shape[0]
    iheight = im.shape[1]
    im = np.lib.pad(im, ((filler, filler), (filler, filler), (0,0)), 'reflect')
    for i in range(filler,iheight+filler,stride):
        for j in range(filler,iwidth+filler,stride):
            im_patch = im[j-filler:j+w+filler, i-filler:i+h+filler, :]
            patch_list.append(im_patch)
    return patch_list
    
def make_patch(X, p_s, stride, filler):
    img_patches = np.asarray([img_crop(X[i], p_s, p_s, stride, filler) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches
    
def make_patch_groundtruth(X, p_s, stride):
    img_patches = np.asarray([crop_groundtruth(X[i], p_s, p_s, stride) for i in range(X.shape[0])])
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
    return img_patches
    
def combine_patch(patches, no_img):
    return patches.reshape(no_img, -1)

def extract_img_features(filename, stride):
    img = ld_img(filename)
    img_patches = img_crop(img, p_s, p_s, stride, filler)
    X = np.asarray([img_patches[i] for i in range(len(img_patches))])
    return X

def output_submission(model, image_filename):
    img_number = int(re.search(r"\d+", image_filename).group(0))
    Xi = ld_img(image_filename)
    Xi = Xi.reshape(1, Xi.shape[0], Xi.shape[1], Xi.shape[2])
    Zi = model.classify(Xi)
    Zi = Zi.reshape(-1)
    p_s = 16
    nb = 0
    print("Processing " + image_filename)
    for j in range(0, Xi.shape[2], p_s):
        for i in range(0, Xi.shape[1], p_s):
            label = int(Zi[nb])
            nb += 1
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def render_submission(model, submission_filename, *image_filenames):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in output_submission(model, fn))
