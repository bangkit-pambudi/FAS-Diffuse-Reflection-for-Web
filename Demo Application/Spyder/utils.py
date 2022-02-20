'''======================= Preprocessing ======================
   Author : Bagas Bangkit Pambudi
   Python 3.7.11
'''

#%% LIBRARY
import numpy as np
import cv2
import dlib

#%% Model Landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#%% Croping Face
def Crop_Face(image):
    '''
    Memotong gambar sesuai dengan 7 titik wajah pada face landmark
    sehingga mendapatkan area Wajah
    '''
    try:
      rect = detector(image)[0]
    except (ValueError,IndexError):
      print("Not Found Face!!")
      return image
  
    sp = predictor(image, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    x = []
    y_alis = []
    y = []
    w = []
    h = []
    
    x.append(landmarks[1][0])
    y_alis.append(landmarks[17][1])
    y_alis.append(landmarks[18][1])
    y_alis.append(landmarks[23][1])
    y_alis.append(landmarks[24][1])
    w.append(landmarks[15][0])
    h.append(landmarks[8][1])
    y.append(min(y_alis))
    
    
    crop_img = image[y[0]:h[0], x[0]:w[0]]
  
    return crop_img

#%% Croping Eye
def Crop_Eye(image):
    '''
    Memotong gambar sesuai dengan 8 titik mata pada face landmark
    sehingga mendapatkan area Wajah
    '''
    
    crop_img = image
  
    return crop_img

#%% Preprocessing
def Preprocessing (base_image,sig,size_x,size_y):
    '''
    Konversi Gray,Memotong daerah Wajah, Gaussian Blur, Resize
    '''
    global face_pre,crop_img
    if(base_image is not None):
      grey = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
      #for (x, y, w, h) in faces:
        #crop_img = grey[y:y + h, x:x + w]
      #crop_img = grey[faces[0,1]:faces[0,1]+faces[0,3], faces[0,0]:faces[0,0]+faces[0,3]]
      crop_img = Crop_Face(grey)
      face_pre = cv2.GaussianBlur(crop_img, ksize=(0, 0), sigmaX=sig, borderType=cv2.BORDER_REPLICATE)
      face = np.double(cv2.resize(np.array(face_pre),(size_x,size_y)))
    
    return face

#%% Feature
def Feature(flash,background):
    '''
    Ektrasi Feature Pantulan
    '''
    
    a = flash - background
    b = flash + background
    c = a/b
    trans = np.transpose(c)
    feat_vec = np.reshape(trans, (trans.size,))
    feat_vec = np.nan_to_num(feat_vec)
    return feat_vec


