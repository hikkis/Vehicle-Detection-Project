
# coding: utf-8

# In[1]:


#Defining utility functions
import glob
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from ExtratFeatures import FeaturesExtract
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# a function to extract features from a list of images
class Window_Sliding:
    def __init__(self, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        

        self.svc=svc
        self.X_scaler=X_scaler 
        self.orient=orient
        self.pix_per_cell=pix_per_cell 
        self.cell_per_block=cell_per_block 
        self.spatial_size=spatial_size
        self.hist_bins=hist_bins
    
    def convert_color(self,img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif conv == 'RGB2YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    def find_cars(self,img, ystart, ystop,scale):

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]  # sub-sampling
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YUV')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient*self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        #nblocks_per_window = (window // pix_per_cell)-1

        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        FE=FeaturesExtract(color_space='YUV', spatial_size=self.spatial_size,
                        hist_bins=self.hist_bins, orient=self.orient,
                        pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True)

        hog1 = FE.get_hog_features(img=ch1, vis=False, feature_vec=False)
        hog2 = FE.get_hog_features(img=ch2,vis=False, feature_vec=False)
        hog3 = FE.get_hog_features(img=ch3,vis=False, feature_vec=False)

        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = FE.bin_spatial(img=subimg, size=self.spatial_size)
                hist_features = FE.color_hist(img=subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.X_scaler.transform(test_stacked)    
                #test_features = scaler.transform(np.array(features).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                    bboxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))

        return draw_img, bboxes

    def apply_sliding_window(self,image):
        bboxes = []
        
        out_img, bboxes1 = self.find_cars(img=image, ystart=400, ystop=500, scale=1.0)
        
        out_img, bboxes2 = self.find_cars(img=out_img, ystart=400, ystop=500, scale=1.3)
       
        out_img, bboxes3 = self.find_cars(img=out_img, ystart=410, ystop=500, scale=1.4)
  
        out_img, bboxes4 = self.find_cars(img=out_img, ystart=420, ystop=556, scale=1.6)

        out_img, bboxes5 = self.find_cars(img=out_img, ystart=430, ystop=556, scale=1.8)
 
        out_img, bboxes6 = self.find_cars(img=out_img, ystart=430, ystop=556, scale=2.0)
 
        out_img, bboxes7 = self.find_cars(img=out_img, ystart=440, ystop=556, scale=1.9)
 
        out_img, bboxes8 = self.find_cars(img=out_img, ystart=400, ystop=556, scale=1.3)
    
        out_img, bboxes9 = self.find_cars(img=out_img, ystart=400, ystop=556, scale=2.2)
   
        out_img, bboxes10 = self.find_cars(img=out_img, ystart=500, ystop=656, scale=3.0)
        bboxes.extend(bboxes1)
        bboxes.extend(bboxes2)
        bboxes.extend(bboxes3)
        bboxes.extend(bboxes4)
        bboxes.extend(bboxes5)
        bboxes.extend(bboxes6)
        bboxes.extend(bboxes7)
        bboxes.extend(bboxes8)
        bboxes.extend(bboxes9)
        bboxes.extend(bboxes10)

        return out_img, bboxes

