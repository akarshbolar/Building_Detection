from helpers import *
from cnn_model import CnnModel
from helpers import *
from keras import backend as K
from sys import exit

if K.backend() != 'theano':
    print('Error: Please change to THEANO BACKEND and continue')
    exit()

K.set_image_dim_ordering('th')

model = CnnModel()
model.load('weights.h5')

model.model.summary()

submission_filename = 'submission.csv'
image_filenames = []
for i in range(1, 2):
    image_filename = 'testing/test_'+str(i)+'/test_' + str(i) + '.png'
    image_filenames.append(image_filename)
    

render_submission(model, submission_filename, *image_filenames)
