from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


data=[[0 for x in range(6)] for y in range(61)]# The Features reading data 68 data
target=[0 for x in range(61)]# 6 labels 6face types

def append_data(points,label):#points ===.68,,,,,label===>shape of the image

    points6=[0 for x in range(6)]

    points6[0]=points[3][1]
    points6[1]=points[4][1]
    points6[2]=points[5][1]
    points6[3]=points[6][1]
    points6[4]=points[7][1]
    points6[5]=points[8][1]

    d1=points6[5] - points6[0]
    d2=points6[5] - points6[1]
    d3=points6[5] - points6[2]
    d4=points6[5] - points6[3]
    d5=points6[5] - points6[4]


    pd1=(float(d2)/float(d1))*100
    pd2=(float(d3)/float(d1))*100
    pd3=(float(d4)/float(d1))*100
    pd4=(float(d5)/float(d1))*100  


    pd1=round(pd1,4)#rounding off to 4 decimal phases  
    pd2=round(pd2,4)
    pd3=round(pd3,4)  
    pd4=round(pd4,4)  

    global data
    global target
    global count

        
    data[count][0]=pd1
    data[count][1]=pd2
    data[count][2]=pd3
    data[count][3]=pd4

    target[count]=label

    count=count+1


def save_data():

    for i in range(0,60):

        file.write(str(data[i][0]))
        file.write(',')
        file.write(str(data[i][1]))
        file.write(',')
        file.write(str(data[i][2]))
        file.write(',')
        file.write(str(data[i][3]))
        file.write(',')
        file.write(str(target[i]))
        file.write('\n')

    file.close()
        


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

        append_data(points,j)#j is a label(The Folder)

        
##
##        for (x,y) in points:
##
##            cv2.circle(image,(x,y),1,(0,0,255),-1)
##            cv2.putText(image,str(k),(x-10,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255,1))
##            k = k+1
##            
##        ###*********We Only Need The Points********
##        cv2.rectangle(image,(x1,y1), (x2,y2), (0,250,0),2)
##        cv2.waitKey(1000)
##        cv2.imshow("image",image)


save_data()
        
