from model02 import get_model0,model_parms 
from init_data_02 import prepare_data_generators,list0 #list of All imgs IDs + crop area code

from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.05*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


# Build model
Batch_size_glb = 5
model_parms(batch_size=Batch_size_glb)
#print('batch size upadate ', BATCH_SIZE)
seg_model = get_model0(Input_img_shape=(300,300,3))
seg_model.summary()


seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('seg_model')

train_gen , valid_gen = prepare_data_generators(split_ratio=0.25,batch_size=Batch_size_glb)


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


loss_history = [seg_model.fit_generator(train_gen, 
                             steps_per_epoch=min( int(0.25*len(list0))//Batch_size_glb, 100),
                             epochs=100, use_multiprocessing=False,
                             validation_data = valid_gen,
                             validation_steps = min( int(0.25*len(list0))//Batch_size_glb, 50),
                             callbacks=callbacks_list,
                            workers=1)]


seg_model.load_weights(weight_path)
seg_model.save('full_best_model.h5')



