import numpy as np
import cv2
import os
import sys
import json

from sklearn.model_selection import train_test_split
#train_ids, valid_ids = train_test_split(l0, test_size = 0.25)

cwd = os.getcwd()
#sys.path.append(os.path.join(cwd, 'Mask_RCNN-master'))
#os.chdir(os.path.join(cwd, 'Mask_RCNN-master'))
'''
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
'''
#os.chdir(cwd)
# initilisation : woring directory and other.
cwd = os.getcwd() #current working dir
map_base_dir = os.path.join(cwd, 'annotations')#D:\C is full\jan py 2021\seg_test02\annotations
map_img_dir = os.path.join(cwd, 'raw')

print(' CWD : ',cwd)
print('annotations Dir : ',map_base_dir)
print('imgs dir Dir    : ',map_img_dir)

def num_to_imge_name(image_number):
    '''
    input integer between  0-85
    85 is total number of images in image folder
    '''
    if image_number in range(0,85):
        img_name0 = str(image_number)
        if img_name0[-1]==9:
            raise Exception("Sorry, image number not available (x_9.png not found)")

        if len(img_name0) ==1:
            img_name0 = '0_'+img_name0 + '.png'
        else:
            img_name0 = img_name0[0]+'_'+img_name0[1]+'.png'
    else:
        print('image number is out of boundaries 0-85')
        print('only 85 images where available when this program is made')
        raise Exception("Sorry, image number should be in range 0-85")

    return img_name0



class masks_data():
    def __init__(self,img_name,path_dir =map_base_dir):
        
        file_name = img_name+'-annotated.json'
        self.json_path = os.path.join(path_dir, file_name)
        self.img_path = os.path.join(map_img_dir, img_name)

        f = open(self.json_path)
        data = json.load(f)
        data = data['labels']

    def list_of_polygons(self,target_class):
        '''
        input : image name ex:'0_0.png',
                target building (): 'Sheds/Garages' or 'Buildings' or 'Houses'
        output : list of polygons in the entire image
                & and their respective color
        '''
        def hex_to_rgb(clr):
            '''internal function to convert color from Hex to (r,g,b)'''
            #print('RGB =', tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
            clr0 = clr.lstrip('#')
            clr_rgb =  tuple(int(clr0[i:i+2], 16) for i in (0, 2, 4))

            return clr_rgb[::-1]#rgb -> bgr for cv2

        f = open(self.json_path)
        data = json.load(f)
        data = data['labels']


        list_poly = []
        clr_list= []

        for label_t in data:
            if label_t['name']==target_class:
                clr_list.append(hex_to_rgb( label_t['color'] )  )
                annotations_t = label_t['annotations']
                for ann in annotations_t:
                    list_poly.append(ann['segmentation'])
        #close the json file after finishing uploading data
        f.close()
        #data_ann = data_an[2]['annotations'][0]['segmentation']
        return list_poly , clr_list

    def label_img(self,label_list=['Sheds/Garages', 'Buildings', 'Houses'],b_mask=True):
        '''
        input : you can specify a list of labels : ['Sheds/Garages', 'Buildings', 'Houses']
                b_mask : Bool :wheter you want to get a binary mask or display output image with colored polygons
        output : binary image mask for all instances OR RGB img output meta 
        '''

        f = open(self.json_path)
        data = json.load(f)
        w = data['width']
        h = data['height']
        f.close()

        img0 = np.zeros((h,w), np.uint8)
        img0.fill(255)

        if b_mask==False:
            img0 = cv2.imread(self.img_path)
        #img44=img33.copy()

        #thickness = 6
            

        for cl in label_list:
            poly_list,clr_list0 = self.list_of_polygons(cl)
            for polys in poly_list:
                #clr0= clr_list0[dummy_cntr]
                poly1 = self.prep_polygone(polys)
                if b_mask == True:
                    img0 = cv2.fillPoly(img0, [poly1], (0,0,0))
                else:
                    color = clr_list0[0]
                    img0 = cv2.polylines(img0, [poly1], True, color, 6)
        return img0


    
    def label_mask_per_instance(self,label_list=['Sheds/Garages', 'Buildings', 'Houses']):
        '''
        input : you can specify a list of labels : ['Sheds/Garages', 'Buildings', 'Houses']
        output : - n-layered binary image. Each layer is mask for one instance. 
                -  a label list  (one lable per layer)
                -  list of bounding polygons
        
        '''

        f = open(self.json_path)
        data = json.load(f)
        w = data['width']
        h = data['height']
        f.close()
        thickness = -1
    
        # extended mask
        #mask_layers = []
        polygon_lists = []
        labels = []
        lbl= 0 # 1 shed ; 2 building ; 3 houses
        for cl in label_list:
            poly_list,_= self.list_of_polygons(cl)
            lbl+=1
            for polys in poly_list:
                #clr0= clr_list0[dummy_cntr]
                poly1 = self.prep_polygone(polys)

                #mask_layers.append(poly1)
                polygon_lists.append(poly1)
                labels.append(lbl)


        
        img0 = np.zeros((h,w,len(polygon_lists)), np.uint8)
        #img0 = np.zeros((h,w,len(mask_layers)), np.uint8)
        dum_img = np.zeros((h,w), np.uint8)
        dum_img.fill(255)
        img0.fill(255)

        for i in range(0,img0.shape[2]):
            img0[:,:,i] = cv2.fillPoly(dum_img, [polygon_lists[i]], (0,0,0))
            #img0[:,:,i] = cv2.fillPoly(dum_img, [mask_layers[i]], (0,0,0))

        return img0 , labels , polygon_lists #, mask_layers



    @staticmethod
    def prep_polygone(seg_list):
        '''convert list of corrdinate into pairs'''
        pts = []

        for i in range(0,len(seg_list),2):
            pts.append( [ round(seg_list[i]) , round(seg_list[i+1]) ] )
        #print('pts here ',pts)
        pts = np.array(pts)
        pts = pts.reshape((-1,1,2))
        pts=np.squeeze(pts)

        return pts


#img_number = 20

#img_name = num_to_imge_name(img_number)

#polys = masks_data(img_name)

#res,_,_ = polys.label_mask_per_instance( )


#print(img_name)
##print(res.shape)
#res=cv2.resize(res,(900,900))
#res0 = res[400:700,400:700]
#res2=polys.label_img()
#cv2.imshow('binary mask ',res2)
#cv2.waitKey(0)

print('here')

class input_processing():
    def __init__(self, im_size=900,crop_size=300):
        '''Assuming the image will be resized into square ex 900x900
        input : - image_size , cropping square size 
        -> create a cropping table generator
        '''
        #self.w,self.h = resize_tuple
        self.im_size = im_size
        self.crop_size = crop_size
        '''
        crop_tab_gen = {}
        k=0

        for i in range(int(im_size/crop_size)):
            for j in range(int(im_size/crop_size)):
                crop_tab_gen[k]=[ crop_size*i , crop_size*(i+1) , crop_size*j , crop_size*(j+1) ]
                k+=1
                
        self.crop_tab_gen = crop_tab_gen
        '''
        self.crop_tab_gen = self.init_gen_crop_img(im_size,crop_size)

    def im_id_crop(self):
        '''
        generating a full list of image IDs
        with a binded reference of the crop area 
        '''
        
        im_names_ext = []
        genarator = self.crop_tab_gen
        n_crops = len( list(genarator.values()) )

        for filename in os.listdir(map_base_dir):
            for i in range(0,n_crops):
                if filename.endswith('json'):
                    im_names_ext.append(filename[0:7]+str(i))
        return im_names_ext

    def croped_gen(self,pic_id):
        '''
        input : img id (name+ref_crop_area)
        output : cropped input image , binary mask cropped
        
        '''
        crop_id = int(pic_id[-1])
        im_name = pic_id[:-1]
        #print(crop_id)
        #print(im_name)

        #full img
        img_path = os.path.join(map_img_dir, im_name)
        img0 = cv2.imread(img_path)
        #binary full img
        polys = masks_data(im_name)
        bin_im = polys.label_img(b_mask=True)

        img0 =   cv2.resize(img0,(self.im_size,self.im_size))
        bin_im = cv2.resize(bin_im,(self.im_size,self.im_size))

        #slicing the image
        gen = self.crop_tab_gen
        crop_tab = gen[crop_id]
        res_img = img0  [ crop_tab[0]:crop_tab[1], crop_tab[2]:crop_tab[3]  ]
        res_bin = bin_im[ crop_tab[0]:crop_tab[1], crop_tab[2]:crop_tab[3]  ]

        #print(res_img.shape)
        #print(bin_im.shape) 

        return res_img, res_bin

    #def augment_gen(self,pic_id)
    

    @staticmethod
    def init_gen_crop_img(im_size,crop_size):

        crop_tab_gen = {}
        k=0

        for i in range(int(im_size/crop_size)):
            for j in range(int(im_size/crop_size)):
                crop_tab_gen[k]=[ crop_size*i , crop_size*(i+1) , crop_size*j , crop_size*(j+1) ]
                k+=1
                
        
        return crop_tab_gen
    


imgs = input_processing(900,300)
gen = imgs.crop_tab_gen
list0 = imgs.im_id_crop()
print('len of imgs ID after cropping ',len(list0))

#rgb_im, bin_im = imgs.croped_gen(list0[5])

#cv2.imshow("test 0",bin_im)
#cv2.waitKey(0)

#print(list0)
from keras.preprocessing import image

def batch_img_gen(in_df, batch_size,img_resize=900,crop_w=300):
    all_groups = in_df
    img_editor = input_processing(im_size= img_resize,crop_size= crop_w)
    out_img, out_seg = [], []
    while True:
        for im_id in np.random.permutation(all_groups):
            
            img_orig,binary_mask = img_editor.croped_gen(im_id)
            #img_orig,binary_mask = croped_gen(im_id,900,300)
            out_img += [img_orig]
            out_seg += [np.expand_dims(binary_mask,-1)]
            if len(out_img)>=batch_size:
                yield (np.stack(out_img, 0)/255.0).astype(np.float32), (np.stack(out_seg, 0)/255.0).astype(np.float32)
                out_img, out_seg = [], []



def batch_augmented_img_gen(in_df, batch_size,img_resize=900,crop_w=300):
    all_groups = in_df
    img_editor = input_processing(im_size= img_resize,crop_size= crop_w)
    out_img, out_seg = [], []
    #augmentation param :
    seed =42
    BATCH_A_SIZE = 2*batch_size

    while True:
        for im_id in np.random.permutation(all_groups):
            
            img_orig,binary_mask = img_editor.croped_gen(im_id)
            #img_orig,binary_mask = croped_gen(im_id,900,300)
            out_img += [img_orig]
            out_seg += [np.expand_dims(binary_mask,-1)]
            if len(out_img)>=batch_size:
                X,Y = (np.stack(out_img, 0)/255.0).astype(np.float32), (np.stack(out_seg, 0)/255.0).astype(np.float32)
                # augmentation :
                
                image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
                mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

                image_datagen.fit(X, augment=True, seed=seed)
                mask_datagen.fit(Y, augment=True, seed=seed)

                x=image_datagen.flow(X,batch_size=BATCH_A_SIZE,shuffle=True, seed=seed)
                y=mask_datagen.flow(Y,batch_size=BATCH_A_SIZE,shuffle=True, seed=seed)
                for i in range (BATCH_A_SIZE):
                    im_batch0 = x.next()
                    imb_batch0 = y.next()
                    im0 = im_batch0[0].astype('uint8')
                    imb0 = imb_batch0[0].astype('uint8')
                    
                    out_img += [im0]
                    out_seg += [np.expand_dims(imb0,-1)]
                yield (np.stack(out_img, 0)/255.0).astype(np.float32), (np.stack(out_seg, 0)/255.0).astype(np.float32)
                out_img, out_seg = [], []
                #yield X,y




def augmented_img_gen(in_df, batch_size,img_resize=900,crop_w=300,is_train=True,batch_A_size=5):
    '''Load the entire dataset and tehn apply data augmentation to produce a well fit img_genrator'''
    # load all imgs
    valid_gen_ed = batch_img_gen(train_ids,len(in_df))
    X_train, Y_train = next(valid_gen_ed)

    # Creating the training Image and Mask generator
    if is_train:            
        image_datagen= image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
        mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    else :
        image_datagen= image.ImageDataGenerator()
        mask_datagen = image.ImageDataGenerator()
    # Keep the same seed for image and mask generators so they fit together
    seed = 42
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)

    x=image_datagen.flow(X_train,batch_size=batch_A_size,shuffle=True, seed=seed)
    y=mask_datagen.flow(Y_train,batch_size=batch_A_size,shuffle=True, seed=seed)

    return zip(x,y)




print('finish dataset prepareing')


from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(list0, test_size = 0.25)

def prepare_data_generators(split_ratio=0.25,batch_size=5,im_id=list0):
    '''split_ratio represents the portion of test set
        input : split ration, batch_size
        output : (x,y) Train_generator , (x,y) Validation Generator

    '''

    from sklearn.model_selection import train_test_split
    train_ids, valid_ids = train_test_split(list0, test_size = 0.25)

    #load the entire dataset (not optimal, should be improved easily but deadline...)
    data_gen = batch_img_gen(train_ids,len(train_ids))
    X_train, Y_train = next(data_gen)
    data_gen = batch_img_gen(valid_ids,len(valid_ids))
    X_val, Y_val = next(data_gen)
    
    print('x set : ', X_train.shape, X_train.dtype, X_train.min(), X_train.max())
    print('y set : ', Y_train.shape, Y_train.dtype, Y_train.min(), Y_train.max())

    #define data augmentation generator
    seed =42
    image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()
    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)

    x=image_datagen.flow(X_train,batch_size=batch_size,shuffle=True, seed=seed)
    y=mask_datagen.flow(Y_train,batch_size=batch_size,shuffle=True, seed=seed)

    image_datagen_val.fit(X_val, augment=True, seed=seed)
    mask_datagen_val.fit(Y_val, augment=True, seed=seed)

    x_val = image_datagen_val.flow(X_val,batch_size=batch_size,shuffle=True, seed=seed)
    y_val = mask_datagen_val.flow(Y_val,batch_size=batch_size,shuffle=True, seed=seed)
    
    
    print('finished dataset prepareing ::: datagenerator are set')
    return zip(x,y) , zip(x_val,y_val)

    
#train, test0 = prepare_data_generators()
