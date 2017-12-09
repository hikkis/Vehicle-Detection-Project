
# coding: utf-8

# In[19]:


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
from skimage.feature import hog
get_ipython().run_line_magic('matplotlib', 'inline')
from ExtratFeatures import FeaturesExtract
from Model import Model_SVM
from Window_Sliding import Window_Sliding
from heatmap import heatmap


# In[20]:


#Collecting data
imagens = glob.glob('./database/non-vehicles/*/*/*.png')
imagevs = glob.glob('./database/vehicles/*/*/*.png')
cars = []
notcars = []
all_cars = []
all_notcars = []

for imagen in imagens:
    all_notcars.append(imagen)
for imagev in imagevs:
    all_cars.append(imagev)

#for image in images:
#    if 'nonvehicle' in image:
#        all_notcars.append(image)
#    else:
#        all_cars.append(image)

# Get only 1/5 of the training data to avoid overfitting
for ix, notcar in enumerate(all_notcars):
    if ix % 5 == 0:
        notcars.append(notcar)

for ix, car in enumerate(all_cars):
    if ix % 5 == 0:
        cars.append(car)

car_image = mpimg.imread(cars[5])
notcar_image = mpimg.imread(notcars[0])

def compare_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

compare_images(car_image, notcar_image, "Car", "Not Car")


# In[21]:


#Define the parameters
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

converted_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
car_ch1 = converted_car_image[:,:,0]
car_ch2 = converted_car_image[:,:,1]
car_ch3 = converted_car_image[:,:,2]

converted_notcar_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YUV)
notcar_ch1 = converted_notcar_image[:,:,0]
notcar_ch2 = converted_notcar_image[:,:,1]
notcar_ch3 = converted_notcar_image[:,:,2]



# Extracting features(Class: ExtratFeatures)
FE = FeaturesExtract(color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

cars_features=FE.features(imgs=cars)
notcars_features=FE.features(imgs=notcars)

car_hog_feature, car_hog_image = FE.get_hog_features(img=car_ch1,
                                        vis=True, feature_vec=True)
notcar_hog_feature, notcar_hog_image = FE.get_hog_features(img=notcar_ch1,
                                        vis=True, feature_vec=True)
# Extracting features(Class: ExtratFeatures)



car_ch1_features = cv2.resize(car_ch1, spatial_size)
car_ch2_features = cv2.resize(car_ch2, spatial_size)
car_ch3_features = cv2.resize(car_ch3, spatial_size)
notcar_ch1_features = cv2.resize(notcar_ch1, spatial_size)
notcar_ch2_features = cv2.resize(notcar_ch2, spatial_size)
notcar_ch3_features = cv2.resize(notcar_ch3, spatial_size)

def show_images(image1, image2, image3, image4,  image1_exp="Image 1", image2_exp="Image 2", image3_exp="Image 3", image4_exp="Image 4"):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    ax3.imshow(image3)
    ax3.set_title(image3_exp, fontsize=20)
    ax4.imshow(image4)
    ax4.set_title(image4_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

show_images(car_ch1, car_hog_image, notcar_ch1, notcar_hog_image, "Car ch 1", "Car ch 1 HOG", "Not Car ch 1", "Not Car ch 1 HOG")    
show_images(car_ch1, car_ch1_features, notcar_ch1, notcar_ch1_features, "Car ch 1", "Car ch 1 features", "Not Car ch 1", "Not Car ch 1 features")    
show_images(car_ch2, car_ch2_features, notcar_ch2, notcar_ch2_features, "Car ch 2", "Car ch 2 features", "Not Car ch 2", "Not Car ch 2 features")    
show_images(car_ch3, car_ch3_features, notcar_ch3, notcar_ch3_features, "Car ch 3", "Car ch 3 features", "Not Car ch 3", "Not Car ch 3 features")


# In[22]:


X = np.vstack((cars_features, notcars_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')



#Use the model of SVM(Class: Model)
Model = Model_SVM(X=scaled_X, y=y)
Fvl,t,t2,score,t3,model= Model.LinearSVC()
#Use the model of SVM(Class: Model)



print('Feature vector length:', Fvl)
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(score, 4))
# Check the prediction time for a single sample


# In[23]:


image1 = mpimg.imread('./test_series/series1.jpg')
image2 = mpimg.imread('./test_series/series2.jpg')
image3 = mpimg.imread('./test_series/series3.jpg')
image4 = mpimg.imread('./test_series/series4.jpg')
image5 = mpimg.imread('./test_series/series5.jpg')
image6 = mpimg.imread('./test_series/series6.jpg')



#Use Window_Sliding to position the cars.(Class: Window_Sliding)
WS = Window_Sliding(svc=model, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
               spatial_size=spatial_size, hist_bins=hist_bins)

output_image1, bboxes1 = WS.apply_sliding_window(image1)
output_image2, bboxes2 = WS.apply_sliding_window(image2)
output_image3, bboxes3 = WS.apply_sliding_window(image3)
output_image4, bboxes4 = WS.apply_sliding_window(image4)
output_image5, bboxes5 = WS.apply_sliding_window(image5)
output_image6, bboxes6 = WS.apply_sliding_window(image6)

image = mpimg.imread('./test_images/test4.jpg')
draw_image = np.copy(image)
output_image, bboxes = WS.apply_sliding_window(image)
#Use Window_Sliding to position the cars.(Class: Window_Sliding)



def show_images(image1, image2, image3,  image1_exp="Image 1", image2_exp="Image 2", image3_exp="Image 3"):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=20)
    ax3.imshow(image3)
    ax3.set_title(image3_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

show_images(output_image1, output_image2, output_image3)
show_images(output_image4, output_image5, output_image6)


# In[43]:


heat = np.zeros_like(output_image[:,:,0]).astype(np.float)
# Add heat to each box in box list
from heatmap import heatmap
HM=heatmap(heatmap=heat)
heat = HM.add_heat(bbox_list=bboxes)

# Apply threshold to help remove false positives
threshold = 1
heat = HM.apply_threshold(threshold=threshold)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = HM.draw_labeled_bboxes(img=np.copy(image), labels=labels)

def show_images(image1, image2,  image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2, cmap='hot')
    ax2.set_title(image2_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

show_images(output_image, heatmap, "Car Positions", "Result")


# In[46]:


#More heatmaps
def get_heatmap(bboxes):
    threshold = 1
    heat = np.zeros_like(output_image[:,:,0]).astype(np.float)
    from heatmap import heatmap
    HM=heatmap(heatmap=heat)
    heat = HM.add_heat(bbox_list=bboxes)
    heat = HM.apply_threshold(threshold=threshold)
    heatmap = np.clip(heat, 0, 255)
    return heatmap

def show_images(image1, image2,  image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=20)
    ax2.imshow(image2, cmap='hot')
    ax2.set_title(image2_exp, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

heatmap1 = get_heatmap(bboxes1)
heatmap2 = get_heatmap(bboxes2)
heatmap3 = get_heatmap(bboxes3)
heatmap4 = get_heatmap(bboxes4)
heatmap5 = get_heatmap(bboxes5)
heatmap6 = get_heatmap(bboxes6)
show_images(output_image1, heatmap1)
show_images(output_image2, heatmap2)
show_images(output_image3, heatmap3)
show_images(output_image4, heatmap4)
show_images(output_image5, heatmap5)
show_images(output_image6, heatmap6)


# In[47]:


#Labeled image
plt.imshow(labels[0], cmap='gray')


# In[48]:


#Resulting bonding boxes
plt.imshow(draw_img)


# In[50]:


#Applying to video
from collections import deque
history = deque(maxlen = 8)

def detect_cars(image):
    WS = Window_Sliding(svc=model, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
               spatial_size=spatial_size, hist_bins=hist_bins)
    bboxes = []
        
    out_img, bboxes1 = WS.find_cars(img=image, ystart=400, ystop=500, scale=1.0)
        
    out_img, bboxes2 = WS.find_cars(img=image, ystart=400, ystop=500, scale=1.3)
       
    out_img, bboxes3 = WS.find_cars(img=out_img, ystart=410, ystop=500, scale=1.4)
  
    out_img, bboxes4 = WS.find_cars(img=out_img, ystart=420, ystop=556, scale=1.6)

    out_img, bboxes5 = WS.find_cars(img=out_img, ystart=430, ystop=556, scale=1.8)
 
    out_img, bboxes6 = WS.find_cars(img=out_img, ystart=430, ystop=556, scale=2.0)
 
    out_img, bboxes7 = WS.find_cars(img=out_img, ystart=440, ystop=556, scale=1.9)
 
    out_img, bboxes8 = WS.find_cars(img=out_img, ystart=400, ystop=556, scale=1.3)
    
    out_img, bboxes9 = WS.find_cars(img=out_img, ystart=400, ystop=556, scale=2.2)
   
    out_img, bboxes10 = WS.find_cars(img=out_img, ystart=500, ystop=656, scale=3.0)
    
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

    heat = np.zeros_like(out_img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    
    from heatmap import heatmap
    HM=heatmap(heatmap=heat)
    heat = HM.add_heat(bbox_list=bboxes)

    # Apply threshold to help remove false positives
    threshold = 1
    HM=heatmap(heatmap=heat)
    heat = HM.apply_threshold(threshold=threshold)

    # Visualize the heatmap when displaying    
    current_heatmap = np.clip(heat, 0, 255)
    history.append(current_heatmap)

    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in history:
        heatmap = heatmap + heat

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = HM.draw_labeled_bboxes(img=np.copy(image), labels=labels)

    return draw_img

img = detect_cars(image)
plt.imshow(img)


# In[51]:


import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[53]:


history = deque(maxlen = 8)
output = 'test_result.mp4'
clip = VideoFileClip("test_video.mp4")
video_clip = clip.fl_image(detect_cars)
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')


# In[55]:


history = deque(maxlen = 8)
output = 'test_result_with Lane Line.mp4'
clip = VideoFileClip("output_video.mp4").subclip(0,30)
video_clip = clip.fl_image(detect_cars)
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')

