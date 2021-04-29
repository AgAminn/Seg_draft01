from model02 import get_model0, model_parms
from init_data_02 import masks_data, input_processing, cwd, map_base_dir,map_img_dir, num_to_imge_name, batch_img_gen

import numpy as np
import cv2
import os


class prediction_seg_msdl():
    '''
    input: img_name + optional path
    output : mask prediction ; 
            post porcessing of thepredicted mask ;
            saved jason file ; 
            diplay predicted result
    '''

    def __init__(self):
        '''load model'''
        Batch_size_glb = 15
        model_parms(batch_size=Batch_size_glb)
        self.seg_model = get_model0(Input_img_shape=(300,300,3))
        model_optim_dir = os.path.join('bkp_models','full_best_model_colab.h5')
        self.seg_model.load_weights(model_optim_dir)

    def mask_predict(self,img_name,imgs_direcotory=map_img_dir):
        
        self.img_name = img_name
        #full img
        img_path = os.path.join(imgs_direcotory, img_name)
        print('image path : ',img_path)
        img0 = cv2.imread(img_path)
        #cv2.imshow('test0000000',img0)
        #cv2.waitKey(0)
        img0 = cv2.resize(img0,(900,900))
        img_out = []

        img_c = img0[0:300,300:600]
        #cv2.imshow('test',img_c)
        img_out+=[img_c]
        img_c2= img0[0:300,600:900]
        img_out+=[img_c2]


        img_out_f = (np.stack(img_out, 0)/255.0).astype(np.float32)
        #(np.stack(out_img, 0)/255.0).astype(np.float32)
        


        #w_l, h_l = []
        imgs_out = []
        for i in range (3):
            for j in range(3):
                img_c =img0[i*300:(i+1)*300,j*300:(j+1)*300]
                imgs_out +=[img_c]
        imgs_out_f = (np.stack(imgs_out, 0)/255.0).astype(np.float32)

        imgs_b_out = []
        
        
        for im_cnt in range(imgs_out_f.shape[0]):
            #predict binary images
            im_po = np.expand_dims(imgs_out_f[im_cnt],axis=0)
            im_po = self.seg_model.predict(im_po)#(im_po,verbose=1)
            im_p = np.squeeze(im_po)

            #remove the accidental padding (woops)
            im_p = im_p[16:284,16:284]
            im_p = cv2.resize(im_p,(300,300))
            imgs_b_out+=[im_p]

        final_img_P = self.stitch_imgs(imgs_b_out)

        return final_img_P#imgs_out_f#img_out_f
    
    def post_processing(self,img,to_save='False'):
        '''

        '''
        #kernel = np.ones((3,3),np.uint8)
        #img = cv2. erode(img,kernel,iterations=4)
        kernel = np.ones((4,4),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        img = cv2.medianBlur(img, 5)
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=5)
        opening = cv2.normalize(opening,opening,0,255,cv2.NORM_MINMAX)
        #opening = np.uint8(opening)

        #ret, opening = cv2.threshold(opening, 200, 255, cv2.THRESH_BINARY)
        

        #cv2.imshow('opening ',opening)#,cv2.waitKey(0)
        #cv2.imshow('closning ',closing)#,cv2.waitKey(0)
        

        res_f = opening.astype(np.uint8)
        res_f= cv2.Canny(res_f , 100, 200)


        return res_f
    
    def save_results_to_json_file(self,predicted_mask):
        improved_im = self.post_processing(self,predicted_mask)
        #rescale rsults
        res_to_scale = cv2.resize(improved_im,(1500,1500))

        # find contours / polygons
        contours, _ = cv2.findContours(res_to_scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_list = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            #seg_arr = {}
            id_n = 0
            if len(hull)> 12:
                polygon_list.append(hull)
        
        import json
        fileJ = self.to_jason_file(image_improved = res_f, im_name = self.img_name)
        with open( os.path.join("predicted_json_files",self.img_name+"_P_annotations.json"), "w") as outfile:
            json.dump(fileJ, outfile)




    @staticmethod
    def stitch_imgs(list_imgs):
    
        im_batch1 =cv2.hconcat([list_imgs[0],list_imgs[1],list_imgs[2]])

        im_batch2 =cv2.hconcat([list_imgs[3],list_imgs[4],list_imgs[5]])

        im_batch3 =cv2.hconcat([list_imgs[6],list_imgs[7],list_imgs[8]])

        im_final = cv2.vconcat([im_batch1,im_batch2,im_batch3])

        
        return im_final
    
    @staticmethod
    def to_jason_file(image_improved,im_name):

        contours, _ = cv2.findContours(image_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon_list = []

        data={}
        data['file_name']=im_name
        data['labels'] =[]
        data['labels'].append({
            'name':'buildings',
            'annotations' :[]
        })
        id_n = 0
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            seg_arr = {}
            seg_arr['id']=im_name+str(id_n)
            
            if len(hull)> 15:
                id_n += 1
                seg_arr['segmentation']=(np.squeeze(hull)).tolist()
                polygon_list.append(hull)
                (data['labels'][0])['annotations'].append(seg_arr)
        
        return data


if __name__=='__main__':

    

    img_test_name = '0_0.png' # change this 
    img_dir = map_img_dir
    
    #load trained model
    predictor = prediction_seg_msdl()

    #predict img mask
    mask = predictor.mask_predict(img_test_name)

    #post processing of the predicted mask
    image_imporved = predictor.post_processing(mask)
    #cv2.imshow('good results',image_imporved),cv2.waitKey(0)


    #save results
    #predictor.save_results_to_json_file(image_imporved)


