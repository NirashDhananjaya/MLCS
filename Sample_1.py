from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2





file =open('train_data.txt','w')

#Loading Resources

#detector =====> classifier for detecting frontal face
detector = dlib.get_frontal_face_detector()

#to find the 68 points of the face as in the image in in Class project folder
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


img_shapes=["Diamond","Oblong","Oval","Round","Square","Triangle"]

count=0

#0 to 10 10 photos
for num in range (0,10):

    #folders
    for j in img_shapes:
        image = cv2.imread("Face Shapes/"+j+"/"+str(num+1)+".jpeg")
                           # 1st ==== > #("Face Shapes/Diamond/1.jpeg)#Loading Images from the folders
        cv2.waitKey(1500)
        cv2.imshow("image",image)    #Time==>52.07
                           
        image=imutils.resize(image,width=500)
