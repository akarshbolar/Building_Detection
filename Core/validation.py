import numpy as np
from helpers import *
from cross_validation import *

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



#from naive_model import NaiveModel
from cnn_model import CnnModel
#from logistic_model import LogisticModel

#model = NaiveModel()
model = CnnModel()
#model = LogisticModel()



np.random.seed(1) # Ensure reproducibility

# Fast (partial) cross validation
quick_cross_validation(model, groundtruth_imgs, imgs, 4, 1)



np.random.seed(1) # Ensure reproducibility

# Full k-fold cross validation
k_fold_cross_validation(model, groundtruth_imgs, imgs, 4, 1)
