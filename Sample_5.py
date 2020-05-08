import csv #coma separated value
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

    


pd1=[]
pd2=[]
pd3=[]
pd4=[]
train_target=[]

with open('train_data.txt','r') as train_data_file:

    data=csv.reader(train_data_file,delimiter=',')

    for cols in data:

        pd1.append(eval(cols[0])) #eval converts string to float text files sends string
        pd2.append(eval(cols[1]))
        pd3.append(eval(cols[2]))
        pd4.append(eval(cols[3]))
        train_target.append(cols[4])

train_data=[[0 for x in range(4)] for y in range (60)]

for i in range(0,60):
    train_data[i][0]=pd1[i]
    train_data[i][1]=pd2[i]
    train_data[i][2]=pd3[i]
    train_data[i][3]=pd4[i]


from sklearn import neighbors

clsfr=neighbors.KNeighborsClassifier()

clsfr.fit(train_data,train_target)



############################################################

import cv2
import imutils
from imutils import face_utils
capture=cv2.VideoCapture(0)





import dlib

detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#reading the camera returning value as o or 1
#resizing image
#preaprocessing
#gray image and
#points gray image and rect oth position
#points will be converted to numpy array *face_utils.shape_to_np(points)
#we are taking 68 points from the image thats taken in by camera
# cv2.circle(image,(x,y),1,(0,255,0),-1) --- taking a circle of points
#only rect has points (Because when there is no face no pints ne)
while(True):

    ret,image=capture.read()
    image=imutils.resize(image,width=800)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    rect=detector(gray,1)

    if (rect):

        points=predictor(gray,rect[0])          
        points=face_utils.shape_to_np(points)##points brings 68 points of the face

        for (x,y) in points:

            cv2.circle(image,(x,y),1,(0,255,0),-1)

    cv2.imshow("MLCS ProJEct",image)
     #key=cv2.waitKey(1)
    cv2.putText(image,"Press P To precr the Face Type",(10,29),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    key=cv2.waitKey(1)


    if(key&0xFF==ord('a')):
        predict_face(points,image)

    

