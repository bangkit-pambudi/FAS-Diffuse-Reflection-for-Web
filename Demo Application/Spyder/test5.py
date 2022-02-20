import cv2
import imutils
import numpy as np
import screen_brightness_control as sbc
import time
#from sklearn.externals import joblib
import joblib
from utils import Preprocessing,Feature
from matplotlib import pyplot as plt
import dlib
from imutils import face_utils



# define a video capture object
blank_image2 = 255 * np.ones(shape=[2000, 2000, 3], dtype=np.uint8)


vid = cv2.VideoCapture(0)
joblib_file = "SVM_LINEAR_DIFF_2.pkl"
joblib_model = joblib.load(joblib_file)
pred = 2

x = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def bounding_box(image,color,list_id,score_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)

    	(x, y, w, h) = face_utils.rect_to_bb(rect)
    	cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    	cv2.putText(image, list_id + ' ,score:' + str(score_id), (x - 10, y - 10),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    	for (x, y) in shape:
    		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    

def Flash():

    cv2.imshow("a", blank_image2)
    #sbc.set_brightness(0)
    
    time.sleep(0.2)
    
    #bg
    ret, frame = vid.read()
    bg = imutils.resize(frame, width=640)
    cv2.imwrite('bg.jpg', bg)
    #cv2.waitKey(1000)
    
    time.sleep(0.2)
    
    #Flash
    ret, frame = vid.read()
    flash = imutils.resize(frame, width=640)
    cv2.imwrite('flash.jpg', flash)
    time.sleep(0.2)
    
    bg = Preprocessing(bg, 5, 100, 100)
    flash = Preprocessing(flash, 5, 100, 100)
    
    cv2.imwrite('pre_flash.jpg', flash)
    cv2.imwrite('pre_bg.jpg', bg)
    
    pantulan = Feature(flash, bg)
    
    predection = joblib_model.predict([pantulan])
    
    if predection == 1:
        plt.plot(np.squeeze(pantulan),label = "live") 
        plt.legend()
        plt.show()
    elif predection == 0:
        plt.plot(np.squeeze(pantulan),label = "spoof",color = 'red') 
        plt.legend()
        plt.show()
    
    return predection
    

list_id = ['-','bagas','alwi']
score_id = ['-',0.133,0.245]
while(True):
    ret, frame = vid.read()
    frame = imutils.resize(frame, width=640)
    k = cv2.waitKey(1)
    #print(k)
    if k == 113:
        #bg,flash = Flash()
        pred = Flash()
        print(str(pred[0]))
        #sbc.set_brightness(20)
    elif k == 119:
        pred = 2
        x = 0
    elif k == 101:
        x = 1
    elif k == 114:
        x = 2
    elif k == 27:
        break
    else:
        if pred == 1:
            score = 'Live'
            color = (0, 255, 0)
        elif pred == 0:
            score = 'Spoof'
            color = (0, 0, 255)
        else:
            score = '-'
            color = (0, 255, 255)
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, score,
                    #(0, 80), font,
                    #2, color,
                    #4, cv2.LINE_AA)
        
        bounding_box(frame,color,list_id[x],score_id[x])
        cv2.imshow('a', frame)
        cv2.waitkey(1)


  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()