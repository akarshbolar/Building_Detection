import numpy as np
from helpers import *
from cross_validation import *
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

# Load the training set
root_directory = "training/"

image_directory = root_directory + "images/"
files = os.listdirectory(image_directory)
n = len(files)
print("Loading " + str(n) + " images")
imgs = np.asarray([ld_img(image_directory + files[i]) for i in range(n)])

groundtruth_directory = root_directory + "groundtruth/"
print("Loading " + str(n) + " images")
groundtruth_imgs = np.asarray([ld_img(groundtruth_directory + files[i]) for i in range(n)])
#print(imgs,groundtruth_imgs)



#from naive_model import NaiveModel
from cnn_model import CnnModel
#from logistic_model import LogisticModel

#model = NaiveModel()
model = CnnModel()
#model = LogisticModel()


np.random.seed(1) # Ensure reproducibility

model.model.summary()
model.train(groundtruth_imgs, imgs)


# Save weights to disk
model.save('saved_weights.h5')

