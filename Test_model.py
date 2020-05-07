import csv #coma separated value


def predict_face(points,image):

    p3=points[3][1]
    p4=points[4][1]
    p5=points[5][1]
    p6=points[6][1]
    p7=points[7][1]
    p8=points[8][1]

    d1 = p8-p3
    d2 = p8-p4
    d3 = p8-p5
    d4 = p8-p6
    d5 = p8-p7

    pd1=float(d2)/float(d1)*100
    pd2=float(d3)/float(d1)*100
    pd3=float(d4)/float(d1)*100
    pd4=float(d5)/float(d1)*100

    
    #test data
    pd1=round(pd1,4)
    pd1=round(pd2,4)
    pd1=round(pd3,4)
    pd1=round(pd4,4)

    test_data=[[pd1,pd2,pd3,pd4]]
    results=clsfr.predict(test_data)#labels (1 from those six will come here as the result)

    cv2.putText(image,"DETECTED",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.putText(image,"Face TYPE:"+str(results[0]),(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)


    cv2.imshow("OUTPUT",image)
    cv2.waitKey(0)
    

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

    cv2.imshow("LIVE",image)
     #key=cv2.waitKey(1)
    cv2.putText(image,"Press P To precr the Face Type",(10,29),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    key=cv2.waitKey(1)


    if(key&0xFF==ord('a')):
        predict_face(points,image)

    

