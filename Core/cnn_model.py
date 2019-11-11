from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from helpers import *

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

class CnnModel:
    
    def __init__(self):
        """ Construct a CNN classifier. """
        
        self.p_s = 16
        self.w_s = 72
        self.filler = (self.w_s - self.p_s) // 2
        self.initialize()
        
    def initialize(self):
        """ Initialize or reset this model. """
        p_s = self.p_s
        w_s = self.w_s
        filler = self.filler
        nb_classes = 2
        
        #MaxPool Area
        pool_size = (2, 2)

        #Theano and Tensorflow Compatibility
        if K.image_dim_ordering() == 'th':
            input_shape = (3, w_s, w_s)
        else:
            input_shape = (w_s, w_s, 3)

        reg = 1e-6

        self.model = Sequential()

        self.model.add(Convolution2D(64, 5, 5, 
                                border_mode='same',
                                input_shape=input_shape
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, 3, 3, 
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, 
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, 3, 3, 
                                border_mode='same'
                               ))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, W_regularizer=l2(reg)
                            )) # Fully connected layer (128 neurons)
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(nb_classes, W_regularizer=l2(reg)
                            ))
        
    
    def train(self, Y, X):
        p_s = self.p_s
        w_s = self.w_s
        filler = self.filler
        
        print('Training set shape: ', X.shape)
        samples_per_epoch = X.shape[0]*X.shape[1]*X.shape[2]//256
        
        # Padding training images
        X_new = np.empty((X.shape[0],
                         X.shape[1] + 2*filler, X.shape[2] + 2*filler,
                         X.shape[3]))
        Y_new = np.empty((Y.shape[0],
                         Y.shape[1] + 2*filler, Y.shape[2] + 2*filler))
        for i in range(X.shape[0]):
            X_new[i] = pd_img(X[i], filler)
            Y_new[i] = pd_img(Y[i], filler)
        X = X_new
        Y = Y_new
            
        batch_size = 25
        nb_classes = 2
        nb_epoch = 2

        def softmax_categorical_crossentropy(y_true, y_pred):
            return K.categorical_crossentropy(y_pred, y_true, from_logits=True)

        opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
        self.model.compile(loss=softmax_categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        np.random.seed(3)
        
        def generate_minibatch():
            while 1:
                # Generate one minibatch
                X_batch = np.empty((batch_size, w_s, w_s, 3))
                Y_batch = np.empty((batch_size, 2))
                for i in range(batch_size):
                    
                    idx = np.random.choice(X.shape[0]) #random image
                    shape = X[idx].shape
                    
                    # Sample a random window from the image
                    center = np.random.randint(w_s//2, shape[0] - w_s//2, 2)
                    sub_image = X[idx][center[0]-w_s//2:center[0]+w_s//2,
                                       center[1]-w_s//2:center[1]+w_s//2]
                    groundtruth_sub_image = Y[idx][center[0]-p_s//2:center[0]+p_s//2,
                                          center[1]-p_s//2:center[1]+p_s//2]
                    
                   
                    threshold = 0.25
                    label = (np.array([np.mean(groundtruth_sub_image)]) > threshold) * 1
                    # augmentation and slip
                    if np.random.choice(2) == 0:
                        sub_image = np.flipud(sub_image) # Flip vertically
                    if np.random.choice(2) == 0:
                        sub_image = np.fliplr(sub_image)  # Flip horizontally
                    
                    # Random rotation in steps of 90°
                    num_rot = np.random.choice(4)
                    sub_image = np.rot90(sub_image, num_rot)
                    label = np_utils.to_categorical(label, nb_classes) 
                    X_batch[i] = sub_image
                    Y_batch[i] = label
                
                if K.image_dim_ordering() == 'th':
                    X_batch = np.rollaxis(X_batch, 3, 1)
                    
                yield (X_batch, Y_batch)
        #reduces learning rate if accuracy doesnt change much
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
                                        
        stop_callback = EarlyStopping(monitor='acc', min_delta=0.0001, patience=11, verbose=1, mode='auto') 
        #stop when converged
        
        try:
            self.model.fit_generator(generate_minibatch(),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=nb_epoch,
                            verbose=1,
                            callbacks=[lr_callback, stop_callback])
        except KeyboardInterrupt:
            pass

        print('Training completed')
        
    def save(self, filename):
        self.model.save_weights(filename)
        
    def load(self, filename):
        self.model.load_weights(filename)
        
    def classify(self, X):
        img_patches = make_patch(X, self.p_s, 16, self.filler)
        
        if K.image_dim_ordering() == 'th':
            img_patches = np.rollaxis(img_patches, 3, 1)
        
        # Run prediction
        Z = self.model.predict(img_patches)
        Z = (Z[:,0] < Z[:,1]) * 1
        
        return combine_patch(Z, X.shape[0])
        
