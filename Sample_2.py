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
        #cv2.waitKey(1500)
        #cv2.imshow("image",image)    Time==>52.07
                           
        image=imutils.resize(image,width=500)
        #############

        #preprocessing

        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('whatever',gray)
        rect = detector(gray,1)

        #rect is the face
        rect = detector(gray,1)#only the face
        #pointers of the picture
        x1=rect[0].left()
        y1=rect[0].top()
        x2=rect[0].right()
        y2=rect[0].bottom()


        points = predictor(gray,rect[0])#gray image rect 0 row will be predicted
        points = face_utils.shape_to_np(points)  #Converting 68 face points to a numpy array

        k=1

        
        
##
        for (x,y) in points:

            cv2.circle(image,(x,y),1,(0,0,255),-1)
            cv2.putText(image,str(k),(x-10,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255,1))
            k = k+1
            
##        ###*********We Only Need The Points********
        cv2.rectangle(image,(x1,y1), (x2,y2), (0,250,0),2)
        cv2.waitKey(1000)
        cv2.imshow("image",image)


save_data()
        

