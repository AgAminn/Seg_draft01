
import numpy as np

from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from keras.layers.core import Lambda,Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
'''
BLOCK_COUNT = 2
EDGE_CROP = 16
BASE_DEPTH = 16
SPATIAL_DROPOUT = 0.25
GAUSSIAN_NOISE = 0.1
BATCH_SIZE = 2
'''
def model_parms(batch_size=2 ,block_count=2,edge_crop=16, base_depth=16,spatial_dropout=0.25, gaussian_noise=0.1):
    '''
    Default values are already in action
    Recommended : update Batch size
    '''
    global BLOCK_COUNT,EDGE_CROP,BASE_DEPTH,SPATIAL_DROPOUT,GAUSSIAN_NOISE,BATCH_SIZE

    BLOCK_COUNT = block_count
    EDGE_CROP = edge_crop
    BASE_DEPTH = base_depth
    SPATIAL_DROPOUT = spatial_dropout
    GAUSSIAN_NOISE = gaussian_noise
    BATCH_SIZE = batch_size
    return 'Model parameter updated'



def conv_bn(x, filt, dl_rate=(1,1), preblock = False):
    y = layers.Convolution2D(filt, (3, 3), 
                             activation='linear', 
                             padding='same', 
                             dilation_rate=dl_rate,
                            use_bias=False)(x)
    if preblock: return y
    y = layers.BatchNormalization()(y)
    return layers.Activation('elu')(y)


def get_model(img_size=(1,300,300,3), num_classes=4):
    inputs = layers.Input(img_size, name = 'RGB_Input')
    print('input shape ',inputs.shape)
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    #outputs = layers.Conv2D(num_classes, 0, activation="softmax", padding="same")(x)

    d = layers.Convolution2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

    # Define the model
    model0 = models.Model(inputs = [inputs],outputs = [d])
    #model = models.Model(inputs, outputs)
    return model0


def get_unet_model(IMG_HEIGHT=300,IMG_WIDTH=300,IMG_CHANNELS=3):
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 1) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()
    return model


def get_model0(Input_img_shape=(300,300,3)):
    
    def conv_bn(x, filt, dl_rate=(1,1), preblock = False):
        y = layers.Convolution2D(filt, (3, 3), activation='linear', padding='same', 
                                    dilation_rate=dl_rate,use_bias=False)(x)
        if preblock:
            return y
        y = layers.BatchNormalization()(y)
        return layers.Activation('elu')(y)

    #in_layer = layers.Input(t_x.shape[1:], name = 'RGB_Input')
    in_layer = layers.Input(Input_img_shape, name = 'RGB_Input')
    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c = conv_bn(pp_in_layer, BASE_DEPTH//2)
    c = conv_bn(c, BASE_DEPTH//2)
    c = conv_bn(c, BASE_DEPTH)

    skip_layers = [pp_in_layer]
    for j in range(BLOCK_COUNT):
        depth_steps = int(np.log2(Input_img_shape[0])-2)
        d = layers.concatenate(skip_layers+[conv_bn(c, BASE_DEPTH*2**j, (2**i, 2**i), preblock=True)
                                            for i in range(depth_steps)])
        d = layers.SpatialDropout2D(SPATIAL_DROPOUT)(d)
        d = layers.BatchNormalization()(d)
        d = layers.Activation('elu')(d)
        # bottleneck
        d = conv_bn(d, BASE_DEPTH*2**(j+1))
        skip_layers += [c]
        c = d

    d = layers.Convolution2D(1, (1, 1), activation='sigmoid', padding='same')(d)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    seg_model = models.Model(inputs = [in_layer],outputs = [d])
    #seg_model.summary()
    return seg_model

# Free up RAM in case the model definition cells were run multiple times
K.clear_session()

# Build model
#seg_model = get_model((300,300), 2)
#seg_model.summary()
#mdl = get_unet_model()



def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.05*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
'''
##################### training phase ####################
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=10, 
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


#seg_model.load_weights('full_best_model.h5')

valid_gen = batch_img_gen(valid_ids, BATCH_SIZE)
loss_history = [seg_model.fit_generator(batch_img_gen(train_ids, BATCH_SIZE), 
                             steps_per_epoch=min( len(train_ids)//BATCH_SIZE, 100),
                             epochs=100, use_multiprocessing=False,
                             validation_data = valid_gen,
                             validation_steps = min( len(train_ids)//BATCH_SIZE, 50),
                             callbacks=callbacks_list,
                            workers=1)]


seg_model.load_weights(weight_path)
#seg_model.save('full_best_model.h5')
'''


if __name__=='__main__':
    K.clear_session()

    # Build model
    model_parms(batch_size=5)
    print('batch size upadate ', BATCH_SIZE)
    seg_model = get_model0(Input_img_shape=(300,300,3))
    seg_model.summary()