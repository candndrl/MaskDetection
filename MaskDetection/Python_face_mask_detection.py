
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def maske_tespit(frame,faceNet,maskNet):
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),
                              (104.0,177.0,123.0))
    faceNet.setInput(blob)
    detections=faceNet.forward()
    print(detections.shape) 


    yüzler=[]
    yüzkonum=[]
    tahmin=[] 
     
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.5:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            (startX,startY)= (max(0,startX),max(0,startY))
            
            (endX,endY)=(min(w-1,endX),min(h-1,endY))

            yüz=frame[startY:endY,startX:endX]
            yüz=cv2.cvtColor(yüz,cv2.COLOR_BGR2RGB)
            yüz=cv2.resize(yüz,(224,224))
            yüz=img_to_array(yüz)
            yüz=preprocess_input(yüz)

            yüzler.append(yüz)
            yüzkonum.append((startX,startY,endX,endY))

    if len(yüzler)>0:
         yüzler=np.array(yüzler,dtype="float32")
         tahmin= maskNet.predict(yüzler,batch_size=32)

    return (yüzkonum,tahmin)

prototxt=r"C:\Users\can\Desktop\MaskDetection\MaskDetectionModelPack-main\face_detector/deploy.prototxt"        
weight=r"C:\Users\can\Desktop\MaskDetection\MaskDetectionModelPack-main\face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNet(prototxt,weight)

maskNet=load_model("mask_detector.model")

print("[INFO] Program başlatılıyor...")
vs=VideoStream(src=0).start()

while True:
    
    frame=vs.read()
    frame=imutils.resize(frame,width=400)

    (yüzkonum,tahmin)=maske_tespit(frame,faceNet,maskNet)

    for(box,tahmin2) in zip(yüzkonum,tahmin):
        (startX,startY,endX,endY)=box
        (mask,withoutMask)=tahmin2
        label="Maske Var" if mask>withoutMask else "Maske Yok"
        color = (0,255,0) if label=="Maske Var" else (0,0,255)
        label="{}: {:.2f}%".format(label,max(mask,withoutMask))

        cv2.putText(frame,label,(startX,startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)

    cv2.imshow("Maske Tespit",frame)
    key=cv2.waitKey(1)&0XFF

    if key== ord("x"):
        break

cv2.destroyAllWindows()
vs.stop()
