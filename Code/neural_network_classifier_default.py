# IMPORTS
import cv2
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

# GLOBAL VALUESr
image_size=(48,48)
csv_address = '../Dataset/Neural_Network/fer2013.csv'
batch_size = 32
number_of_epochs = 110
input_shape = (48, 48, 1)
verbose = 1
number_of_classes = 7
patience = 50
saved_files_path = 'models/'
L2_regularization=0.01
log_file_path = saved_files_path + '_emotion_training.log'
trained_models_path = saved_files_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

# LOADS ALL THE RESIZED FACES & EMOTIONS 
def load_data():
    data = pd.read_csv(csv_address)
    pixels = data['pixels'].tolist()
    width = 48
    height = 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

#make it normalized between -1 and +1
def normalize_input_data(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
faces, emotions = load_data()
faces = normalize_input_data(faces)
xtrain,xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
data_generator = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,zoom_range=.1,horizontal_flip=True)
regularization = l2(L2_regularization)
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# SECTION 1
remained = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
remained = BatchNormalization()(remained)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, remained])
# SECTION 2
remained = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
remained = BatchNormalization()(remained)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, remained])
# SECTION 3
remained = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
remained = BatchNormalization()(remained)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, remained])
# SECTION 4
remained = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
remained = BatchNormalization()(remained)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, remained])
x = Conv2D(number_of_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax',name='predictions')(x)

model = Model(img_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
# CALL_BACKS
CALL_BACK_csv_logger = CSVLogger(log_file_path, append=False) #log files consists of accuracy etc
CALL_BACK_early_stop = EarlyStopping('val_loss', patience=patience) # when stops that cycle
CALL_BACK_reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1) # when reduce learning rate in that epoch 
CALL_BACK_model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True) # when to save
callbacks = [CALL_BACK_model_checkpoint, CALL_BACK_csv_logger, CALL_BACK_early_stop, CALL_BACK_reduce_lr] #all the callbacks
 
model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),steps_per_epoch=len(xtrain) / batch_size,epochs=number_of_epochs, verbose=1, callbacks=callbacks,validation_data=(xtest,ytest))
