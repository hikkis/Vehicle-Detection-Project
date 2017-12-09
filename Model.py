
# coding: utf-8

# In[3]:


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


# In[4]:


# a function to define the model
class Model_SVM:
    
    def __init__(self, X, y, ):
        self.X=X
        self.y=y
    def LinearSVC(self):
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=rand_state)
        
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        model = svc
        score=svc.score(X_test, y_test)
        
        t3=time.time()
        
        return len(X_train[0]), t, t2, score,t3,model

